# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend Agent Hint free-block eviction policy."""

import time
from dataclasses import dataclass, field

from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)


@dataclass(slots=True)
class AgentHintKVCacheBlock(KVCacheBlock):
    # Sessions currently referencing this block (block → session reverse
    # index). Written by SAM on allocate / cache-hit; discarded on evict;
    # cleared when the block's cached hash is wiped for reuse. SSM never
    # touches this field.
    # Zone-B membership is derived: len(sessions) > 0 keeps the block in
    # zone B (offload does not touch sessions, so offloaded blocks stay
    # anchored until explicitly evicted or physically reused — replacing
    # both the old _is_offload_block flag and the dual-counter scheme;
    # design §7.1 decision: compute len() instead of caching a counter,
    # len(set) is O(1) and no set/counter drift is possible).
    _sessions: set[str] = field(default_factory=set)
    _ttl_expire_at: float = 0.0
    # Bucket index in zone B (-1 when the block is not in zone B).
    # Set by FreeKVCacheBlockQueue when the block is inserted into a B-zone
    # bucket, and reset to -1 when the block leaves zone B.
    _zone_b_bucket: int = -1

    # Index in the C-zone min-heap (-1 when the block is not in the heap).
    # Set by FreeKVCacheBlockQueue when the block enters zone C, and reset
    # to -1 when the block leaves zone C.  Enables O(log n) heap removal.
    _heap_index: int = -1

    @property
    def sessions(self) -> set[str]:
        return self._sessions

    @property
    def num_session_refs(self) -> int:
        return len(self._sessions)

    @property
    def ttl_expire_at(self) -> float:
        return self._ttl_expire_at

    @property
    def is_ephemeral(self) -> bool:
        return self._ttl_expire_at > 0 and time.monotonic() < self._ttl_expire_at

    @property
    def zone_b_bucket(self) -> int:
        return self._zone_b_bucket

    @property
    def heap_index(self) -> int:
        return self._heap_index

    def set_session_state(
        self,
        *,
        ttl_expire_at: float | None = None,
        zone_b_bucket: int | None = None,
        heap_index: int | None = None,
    ) -> None:
        if ttl_expire_at is not None:
            self._ttl_expire_at = ttl_expire_at
        if zone_b_bucket is not None:
            self._zone_b_bucket = zone_b_bucket
        if heap_index is not None:
            self._heap_index = heap_index

    def reset_session_state(self) -> None:
        self._sessions.clear()
        self._ttl_expire_at = 0.0
        self._zone_b_bucket = -1
        self._heap_index = -1


class AgentHintFreeKVCacheBlockQueue(FreeKVCacheBlockQueue):
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        super().__init__(blocks)

        # A zone: allocatable blocks without any session refs
        #         (len(sessions) == 0).
        # B zone: allocatable blocks with session refs — either a current
        #         session ref or a session-aware ref not yet consumed by an
        #         explicit evict.
        # C zone: deadline-protected blocks.
        # Assume all the blocks are in the A zone at first.
        self.zone1_end: KVCacheBlock | None = blocks[-1] if blocks else None
        self.zone2_end: KVCacheBlock | None = None

        # B zone bucketing: blocks are grouped by num_session_refs into
        # buckets.  Bucket k (0 <= k < num_b_buckets - 1) holds blocks with
        # num_session_refs == k.  The last bucket (num_b_buckets - 1) holds
        # blocks with num_session_refs >= num_b_buckets - 1.  Within each
        # bucket, blocks are in FIFO order (insertion order).  Buckets are
        # physically ordered by index in the linked list so that popleft
        # always takes from the lowest-index non-empty bucket, preserving
        # the "least-bad eviction" semantics.
        # zone_b_bucket_ends[k] is the last block in bucket k, or None if
        # the bucket is empty.
        self.num_b_buckets: int = 10
        self.zone_b_bucket_ends: list[KVCacheBlock | None] = (
            [None] * self.num_b_buckets
        )

        # C zone min-heap: blocks ordered by _ttl_expire_at so that the
        # earliest-expiring block is always at the heap root.  This lets
        # lazy_scan_zone_c find expired blocks in O(k log n) instead of
        # scanning the entire C zone linked list.  The linked list still
        # holds the blocks physically (in FIFO insertion order, no longer
        # sorted by TTL), while the heap provides the TTL-ordered index.
        # Each block's _heap_index tracks its position in the list for
        # O(log n) removal when it leaves zone C.
        self.zone_c_heap: list[KVCacheBlock] = []
        # zone_c_end tracks the tail of the C-zone linked list (or None
        # when C zone is empty), enabling O(1) FIFO tail insertion.
        self.zone_c_end: KVCacheBlock | None = None

    # ------------------------------------------------------------------
    # B-zone bucket helpers
    # ------------------------------------------------------------------
    def _get_bucket_index(self, num_session_refs: int) -> int:
        """Compute the B-zone bucket index for a given session ref count.

        Buckets 0 .. num_b_buckets-2 hold blocks with num_session_refs ==
        bucket_index.  The last bucket (num_b_buckets-1) holds blocks with
        num_session_refs >= num_b_buckets-1.
        """
        return min(num_session_refs, self.num_b_buckets - 1)

    def _find_b_bucket_insertion_point(
        self, bucket_idx: int
    ) -> KVCacheBlock:
        """Return the block **after which** a new block should be inserted
        to land at the tail of *bucket_idx*.

        If the bucket is non-empty, returns its current tail (append to
        bucket).  If the bucket is empty, returns the tail of the nearest
        non-empty bucket with a lower index, or ``zone1_end`` /
        ``fake_free_list_head`` when no lower bucket is non-empty.
        """
        if self.zone_b_bucket_ends[bucket_idx] is not None:
            return self.zone_b_bucket_ends[bucket_idx]

        for j in range(bucket_idx - 1, -1, -1):
            if self.zone_b_bucket_ends[j] is not None:
                return self.zone_b_bucket_ends[j]

        return self.zone1_end or self.fake_free_list_head

    def _update_zone2_end(self) -> None:
        """Recompute ``zone2_end`` from bucket tails.

        ``zone2_end`` is always the tail of the highest-index non-empty
        bucket (or None when all buckets are empty).
        """
        for j in range(self.num_b_buckets - 1, -1, -1):
            if self.zone_b_bucket_ends[j] is not None:
                self.zone2_end = self.zone_b_bucket_ends[j]
                return
        self.zone2_end = None

    def _on_b_block_removed(self, block: KVCacheBlock) -> None:
        """Update bucket tail and ``zone2_end`` when *block* leaves zone B.

        Must be called **before** the block's ``prev_free_block`` /
        ``next_free_block`` pointers are modified by the caller, so that
        the previous block can be inspected to find the new bucket tail.
        """
        bucket_idx = block._zone_b_bucket
        if bucket_idx < 0:
            return  # Block was not in zone B — nothing to update.

        if self.zone_b_bucket_ends[bucket_idx] is block:
            # Block was the tail of its bucket; find the new tail by
            # looking at its predecessor in the linked list.
            prev = block.prev_free_block
            if (
                prev is not None
                and prev is not self.fake_free_list_head
                and prev is not self.zone1_end
                and prev._zone_b_bucket == bucket_idx
            ):
                self.zone_b_bucket_ends[bucket_idx] = prev
            else:
                self.zone_b_bucket_ends[bucket_idx] = None

        block._zone_b_bucket = -1
        self._update_zone2_end()

    # ------------------------------------------------------------------
    # C-zone min-heap helpers
    # ------------------------------------------------------------------
    def _heap_sift_up(self, idx: int) -> None:
        """Sift the element at *idx* up to restore min-heap order by
        ``_ttl_expire_at``."""
        heap = self.zone_c_heap
        item = heap[idx]
        item_ttl = item._ttl_expire_at
        while idx > 0:
            parent = (idx - 1) >> 1
            parent_block = heap[parent]
            if parent_block._ttl_expire_at <= item_ttl:
                break
            heap[idx] = parent_block
            parent_block._heap_index = idx
            idx = parent
        heap[idx] = item
        item._heap_index = idx

    def _heap_sift_down(self, idx: int) -> None:
        """Sift the element at *idx* down to restore min-heap order."""
        heap = self.zone_c_heap
        n = len(heap)
        item = heap[idx]
        item_ttl = item._ttl_expire_at
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            smallest = idx
            smallest_ttl = item_ttl
            if left < n and heap[left]._ttl_expire_at < smallest_ttl:
                smallest = left
                smallest_ttl = heap[left]._ttl_expire_at
            if right < n and heap[right]._ttl_expire_at < smallest_ttl:
                smallest = right
                smallest_ttl = heap[right]._ttl_expire_at
            if smallest == idx:
                break
            heap[idx] = heap[smallest]
            heap[idx]._heap_index = idx
            idx = smallest
        heap[idx] = item
        item._heap_index = idx

    def _heap_push(self, block: KVCacheBlock) -> None:
        """Push *block* onto the C-zone min-heap."""
        heap = self.zone_c_heap
        idx = len(heap)
        heap.append(block)
        block._heap_index = idx
        self._heap_sift_up(idx)

    def _heap_pop(self) -> KVCacheBlock:
        """Pop and return the minimum (earliest-expiring) block."""
        heap = self.zone_c_heap
        if not heap:
            raise IndexError("pop from empty C-zone heap")
        top = heap[0]
        last = heap.pop()
        if heap:
            heap[0] = last
            last._heap_index = 0
            self._heap_sift_down(0)
        top._heap_index = -1
        return top

    def _heap_remove(self, block: KVCacheBlock) -> None:
        """Remove *block* from the C-zone min-heap in O(log n).

        Uses the block's ``_heap_index`` to locate it, replaces it with
        the last element, then restores heap order.
        """
        heap = self.zone_c_heap
        idx = block._heap_index
        if idx < 0:
            return  # Block is not in the heap — nothing to remove.
        last = heap.pop()
        if idx < len(heap):
            heap[idx] = last
            last._heap_index = idx
            # Restore order — the displaced element may need to move
            # up or down depending on its TTL vs the removed block's.
            if last._ttl_expire_at < block._ttl_expire_at:
                self._heap_sift_up(idx)
            else:
                self._heap_sift_down(idx)
        block._heap_index = -1

    def _on_c_block_removed(self, block: KVCacheBlock) -> None:
        """Update heap and zone_c_end when *block* leaves zone C.

        Must be called **before** the block's linked-list pointers are
        modified by the caller, so that ``prev_free_block`` can be
        inspected to update ``zone_c_end``.
        """
        self._heap_remove(block)

        if self.zone_c_end is block:
            prev = block.prev_free_block
            if (
                prev is not None
                and prev is not self.fake_free_list_head
                and prev is not self.zone1_end
                and prev is not self.zone2_end
                and prev._heap_index >= 0
            ):
                self.zone_c_end = prev
            else:
                self.zone_c_end = None

    def lazy_scan_zone_c(self, check_ttl) -> None:
        """Lazy TTL expiry: pop expired blocks from the C-zone min-heap and
        promote them to zone A or B.

        Uses the min-heap (ordered by _ttl_expire_at) to find expired blocks
        in O(k log n) where k is the number of expired blocks, instead of
        scanning the entire C-zone linked list in O(n).
        """
        if not check_ttl:
            return

        now = time.monotonic()
        heap = self.zone_c_heap

        # Pop all blocks whose TTL has expired from the heap root.
        # The heap guarantees the earliest-expiring block is at index 0.
        while heap and heap[0]._ttl_expire_at > 0 and heap[0]._ttl_expire_at <= now:
            block = self._heap_pop()
            block._ttl_expire_at = 0.0

            # Remove the block from the C-zone linked list and promote it.
            # _heap_pop already set block._heap_index = -1, so
            # _on_c_block_removed will skip the heap removal; we only need
            # to update zone_c_end before linked-list pointer manipulation.
            if self.zone_c_end is block:
                prev = block.prev_free_block
                if (
                    prev is not None
                    and prev is not self.fake_free_list_head
                    and prev is not self.zone1_end
                    and prev is not self.zone2_end
                    and prev._heap_index >= 0
                ):
                    self.zone_c_end = prev
                else:
                    self.zone_c_end = None

            if block.num_session_refs > 0:
                self.promote_to_zone_b(block)
                logger.debug(
                    f"lazy_scan_zone_c: Promoted expired C-zone block id {block.block_id} "
                    f"to zone B."
                    f"block.num_session_refs: {block.num_session_refs}. "
                    f"block._ttl_expire_at: {block._ttl_expire_at}. "
                )
            else:
                self.promote_to_zone_a(block)
                logger.debug(
                    f"lazy_scan_zone_c: Promoted expired C-zone block id {block.block_id} "
                    f"to zone A."
                    f"block.num_session_refs: {block.num_session_refs}. "
                    f"block._ttl_expire_at: {block._ttl_expire_at}. "
                )

    def popleft(self, check_ttl: bool = True) -> KVCacheBlock:
        """Pop from zone A first, then zone B, and reduce num_free_blocks by 1.
        Zone C is not allocatable unless a block's TTL has expired. When A/B are
        empty, lazily scan C and promote expired blocks to A/B.

        Returns:
            The first free block.
        """
        if (
            self.fake_free_list_head.next_free_block is self.fake_free_list_tail
            or self.fake_free_list_head.next_free_block is None
        ):
            assert self.num_free_blocks == 0, (
                f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                "with the free list."
            )
            raise ValueError("No free blocks available")

        self.lazy_scan_zone_c(check_ttl)

        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block

        if first_block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(
                "Invalid block found in popleft() "
                "which doesn't have a valid next_free_block"
            )

        if first_block is self.zone1_end:
            self.zone1_end = None
        self._on_b_block_removed(first_block)
        self._on_c_block_removed(first_block)

        # Connect fake_head and the next block of first_block (i.e. second block
        # or fake tail).
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head

        # Remove the block from the linked list.
        first_block.prev_free_block = first_block.next_free_block = None

        self.num_free_blocks -= 1
        return first_block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop the first n allocatable free blocks and reduce num_free_blocks by n.

        Args:
            n: The number of blocks to pop.

        Returns:
            A list of n free blocks.
        """
        if n == 0:
            return []
        assert self.num_free_blocks >= n

        only_zone_a = (
            self.zone2_end is None
            and self.zone1_end is not None
            and self.zone1_end is self.fake_free_list_tail.prev_free_block
        )

        if not only_zone_a:
            ret = [self.popleft(check_ttl=True)]
            for _ in range(1, n):
                ret.append(self.popleft(check_ttl=False))
            return ret

        # only zone A，no need to check_ttl。
        self.num_free_blocks -= n

        curr_block = self.fake_free_list_head.next_free_block
        # Pop n blocks from the head of the list
        ret = []

        for _ in range(n):
            assert curr_block is not None
            ret.append(curr_block)
            last_block = curr_block
            curr_block = curr_block.next_free_block
            # Reset prev_free_block and next_free_block of all popped blocks
            last_block.prev_free_block = None
            last_block.next_free_block = None

        if curr_block is not None:
            # The queue is not empty, connect the fake head to
            # the new first block.
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head

        if self.num_free_blocks == 0:
            self.zone1_end = None

        return ret

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.

        Args:
            block: The block to remove.
        """
        if block.prev_free_block is None or block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        prev_block = block.prev_free_block
        next_block = block.next_free_block

        if block is self.zone1_end:
            self.zone1_end = (
                prev_block if prev_block is not self.fake_free_list_head else None
            )

        self._on_b_block_removed(block)
        self._on_c_block_removed(block)
        
        # Link the previous block to the next block.
        prev_block.next_free_block = next_block
        # Link the next block to the previous block.
        next_block.prev_free_block = prev_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Zone A:
            Blocks without any session references (num_session_refs == 0,
            i.e. len(sessions) == 0).

        Zone B:
            Blocks with session references (num_session_refs > 0), grouped
            into buckets by num_session_refs.  Within each bucket, blocks
            are in FIFO order.

        Zone C:
            TTL-protected blocks.  The linked list is in FIFO insertion
            order (no longer sorted by TTL); a min-heap indexed by
            _ttl_expire_at provides O(log n) TTL-ordered access for
            lazy_scan_zone_c.
        """
        if self.fake_free_list_tail.prev_free_block is None:
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist"
            )

        if block.is_ephemeral:
            logger.debug(
                "append: Appending TTL-protected block id %s to zone C.",
                block.block_id,
            )

            # C 区位于 A/B 区之后, 按 FIFO 顺序插入到 C 区尾部.
            # TTL 排序由 min-heap 维护, 链表不再按 TTL 排序.
            prev_block = (
                self.zone_c_end
                or self.zone2_end
                or self.zone1_end
                or self.fake_free_list_head
            )
            # O(log n) heap push for TTL-ordered access.
            self._heap_push(block)
            self.zone_c_end = block

        elif block.num_session_refs > 0:
            logger.debug("append: Appending block id %s to zone B.", block.block_id)

            # B 区使用桶分组: 按 num_session_refs 值分桶, 桶内 FIFO.
            bucket_idx = self._get_bucket_index(block.num_session_refs)
            block._zone_b_bucket = bucket_idx
            prev_block = self._find_b_bucket_insertion_point(bucket_idx)
            self.zone_b_bucket_ends[bucket_idx] = block
            self._update_zone2_end()

        else:
            logger.debug("append: Appending block id %s to zone A.", block.block_id)

            # A 区保持 FIFO 顺序。
            prev_block = self.zone1_end or self.fake_free_list_head
            self.zone1_end = block

        next_block = prev_block.next_free_block
        if next_block is None:
            raise RuntimeError(
                f"Invalid insertion position for block {block.block_id}"
            )

        # Connect the new block after prev_block.
        prev_block.next_free_block = block
        block.prev_free_block = prev_block

        # Connect the next block after the new block.
        block.next_free_block = next_block
        next_block.prev_free_block = block

        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks back into the free list

        Args:
            blocks: The blocks to append.
        """
        if len(blocks) == 0:
            return

        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None, (
            "prev_free_block of fake_free_list_tail should always exist"
        )

        only_zone_a_or_empty = (
            self.num_free_blocks == 0 or self.zone1_end is last_block
        )

        blocks_only_zone_a = all(
            not block.is_ephemeral and block.num_session_refs == 0
            for block in blocks
        )

        if not only_zone_a_or_empty or not blocks_only_zone_a:
            for block in blocks:
                self.append(block)
            return
        
        # Add inter-connections between consecutive blocks
        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        # Connect the last block of <blocks> to the fake tail
        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block

        self.num_free_blocks += len(blocks)

        self.zone1_end = last_block

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.

        Returns:
            A list of free blocks.
        """
        ret = []
        if self.fake_free_list_head.next_free_block is None:
            raise RuntimeError(
                "next_free_block of fake_free_list_head should always exist"
            )
        # Start from the first block
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        # As long as next_free_block is available, we haven't reached to
        # the fake tail yet.
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret
    
    def promote_to_zone_a(self, block: KVCacheBlock) -> None:
        """Move an existing free block to the tail of zone A."""
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"promote_to_zone_a() called on invalid block: {block}")

        if block is self.zone1_end:
            return

        # Remove without changing num_free_blocks.
        prev_block = block.prev_free_block
        next_block = block.next_free_block

        self._on_b_block_removed(block)
        self._on_c_block_removed(block)

        prev_block.next_free_block = next_block
        next_block.prev_free_block = prev_block

        # Insert into A tail.
        prev_block = self.zone1_end or self.fake_free_list_head
        next_block = prev_block.next_free_block
        assert next_block is not None

        prev_block.next_free_block = block
        block.prev_free_block = prev_block
        block.next_free_block = next_block
        next_block.prev_free_block = block
        self.zone1_end = block

    def promote_to_zone_b(self, block: KVCacheBlock) -> None:
        """Move an existing free block into zone B.

        Zone B uses bucket grouping: blocks are placed into buckets by
        ``num_session_refs``.  Within each bucket, blocks are in FIFO order.
        """
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(
                f"promote_to_zone_b() called on invalid block: {block}"
            )

        # Remove without changing num_free_blocks.
        old_prev = block.prev_free_block
        old_next = block.next_free_block

        # block 原来是 A 区最后一个节点。
        if block is self.zone1_end:
            self.zone1_end = (
                old_prev if old_prev is not self.fake_free_list_head else None
            )

        # block 原来是 B 区某个桶中的节点。
        self._on_b_block_removed(block)

        # block 原来是 C 区中的节点 (e.g. lazy_scan 过期后 promote)。
        self._on_c_block_removed(block)

        old_prev.next_free_block = old_next
        old_next.prev_free_block = old_prev

        # B 区使用桶分组: 按 num_session_refs 值分桶, 桶内 FIFO.
        bucket_idx = self._get_bucket_index(block.num_session_refs)
        block._zone_b_bucket = bucket_idx
        prev_block = self._find_b_bucket_insertion_point(bucket_idx)

        next_block = prev_block.next_free_block
        if next_block is None:
            raise RuntimeError(
                f"Invalid zone B insertion position for block {block.block_id}"
            )

        prev_block.next_free_block = block
        block.prev_free_block = prev_block

        block.next_free_block = next_block
        next_block.prev_free_block = block

        self.zone_b_bucket_ends[bucket_idx] = block
        self._update_zone2_end()

        logger.debug(
            f"promote_to_zone_b: Promoting block id {block.block_id} to zone B."
            f"block.num_session_refs: {block.num_session_refs}. "
            f"block._ttl_expire_at: {block._ttl_expire_at}. "
        )
    
    def promote_to_zone_c(self, block: KVCacheBlock) -> None:
        """Move an existing free block into zone C.

        Zone C linked list is in FIFO insertion order; a min-heap indexed
        by _ttl_expire_at provides O(log n) TTL-ordered access.
        """
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(
                f"promote_to_zone_c() called on invalid block: {block}"
            )

        # 先从原位置摘除，但不修改 num_free_blocks。
        old_prev = block.prev_free_block
        old_next = block.next_free_block

        # block 原来是 A 区最后一个节点。
        if block is self.zone1_end:
            self.zone1_end = (
                old_prev if old_prev is not self.fake_free_list_head else None
            )

        # block 原来是 B 区某个桶中的节点。
        self._on_b_block_removed(block)

        # block 原来是 C 区中的节点 (e.g. TTL 更新后重新定位)。
        self._on_c_block_removed(block)

        old_prev.next_free_block = old_next
        old_next.prev_free_block = old_prev

        block.prev_free_block = None
        block.next_free_block = None

        # C 区从 A/B 区之后开始, 插入到 C 区尾部 (FIFO 顺序).
        prev_block = (
            self.zone_c_end
            or self.zone2_end
            or self.zone1_end
            or self.fake_free_list_head
        )

        # O(log n) heap push for TTL-ordered access.
        self._heap_push(block)

        next_block = prev_block.next_free_block
        if next_block is None:
            raise RuntimeError(
                f"Invalid zone C insertion position for block {block.block_id}"
            )

        prev_block.next_free_block = block
        block.prev_free_block = prev_block

        block.next_free_block = next_block
        next_block.prev_free_block = block

        self.zone_c_end = block

        logger.debug(
            f"promote_to_zone_c: Promoting block id {block.block_id} to zone C."
            f"block.num_session_refs: {block.num_session_refs}. "
            f"block._ttl_expire_at: {block._ttl_expire_at}. "
        )
    
    def on_block_meta_changed(self, block: KVCacheBlock) -> None:
        """block metadata（sessions 或 _ttl_expire_at）变化后重新评估分区"""
        if block.ref_cnt > 0 or block.is_null:
            return  # block 被活跃请求使用，不在 free queue 中

        if block.is_ephemeral:
            # C区：ephemeral 保护中，不可分配
            self.promote_to_zone_c(block)
        elif block.num_session_refs > 0:
            # B区：无 ephemeral 保护，但当前 session 引用或
            # session-aware 引用（尚未被显式 evict 消耗）仍存在
            self.promote_to_zone_b(block)
        else:
            # A区：无保护、无引用，优先分配
            self.promote_to_zone_a(block)
