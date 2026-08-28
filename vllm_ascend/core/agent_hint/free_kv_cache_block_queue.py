# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend Agent Hint free-block eviction policy."""

import time

from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock



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

        # A zone: allocatable blocks without session refs.
        # B zone: allocatable blocks with session refs.
        # C zone: deadline-protected blocks.
        # Assume all the blocks are in the A zone at first.
        self.zone1_end: KVCacheBlock | None = blocks[-1] if blocks else None
        self.zone2_end: KVCacheBlock | None = None

    def lazy_scan_zone_c(self, check_ttl: bool) -> None:
        """Promote blocks whose TTL has expired from zone C."""
        if check_ttl:
            # Scan C once and promote blocks whose deadline has expired.
            if self.zone2_end is not None:
                curr_block = self.zone2_end.next_free_block
            elif self.zone1_end is not None:
                curr_block = self.zone1_end.next_free_block
            else:
                curr_block = self.fake_free_list_head.next_free_block

            while curr_block is not self.fake_free_list_tail:
                if curr_block is None or curr_block.next_free_block is None:
                    raise RuntimeError(
                        "Invalid block found in popleft() "
                        "which doesn't have a valid next_free_block"
                    )

                next_block = curr_block.next_free_block

                if (
                    curr_block.ttl_expire_at > 0
                    and time.monotonic() >= curr_block.ttl_expire_at
                ):
                    curr_block.set_session_state(ttl_expire_at=0.0)
                    if curr_block.num_session_refs > 0:
                        self.promote_to_zone_b(curr_block)
                    else:
                        self.promote_to_zone_a(curr_block)

                curr_block = next_block

    def popleft(self, check_ttl: bool = True) -> KVCacheBlock:
        """Pop from zone A first, then zone B, and reduce num_free_blocks by 1.
        Zone C is a soft-protection tier used only after zones A and B.

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
        if first_block is self.zone2_end:
            self.zone2_end = None

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

        if block is self.zone2_end:
            self.zone2_end = (
                prev_block
                if (
                    prev_block is not self.fake_free_list_head
                    and prev_block is not self.zone1_end
                )
                else None
            )

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
            Blocks without session references.

        Zone B:
            Blocks with session references, ordered by num_session_refs ascending.

        Zone C:
            deadline-protected blocks, ordered by _ttl_expire_at ascending.
        """
        if self.fake_free_list_tail.prev_free_block is None:
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist"
            )

        if block.is_ephemeral:
            logger.debug(
                "append: Appending deadline-protected block id %s to zone C.",
                block.block_id,
            )

            # C 区位于 A/B 区之后, 按 _ttl_expire_at 升序排列: 小的靠前, 大的靠后
            prev_block = self.zone2_end or self.zone1_end or self.fake_free_list_head
            curr_block = prev_block.next_free_block

            while curr_block is not self.fake_free_list_tail:
                if curr_block is None:
                    raise RuntimeError("Invalid zone C boundary")

                # 插入到第一个过期时间更大的 block 前面。
                # 使用 >，使相同 TTL 的 block 保持 FIFO 顺序。
                if curr_block.ttl_expire_at > block.ttl_expire_at:
                    break

                prev_block = curr_block
                curr_block = curr_block.next_free_block

        elif block.num_session_refs > 0 or block.is_offload_block:
            logger.debug("append: Appending block id %s to zone B.", block.block_id)

            # B 区按照 session 引用数量升序排列, 引用少的靠前，引用多的靠后
            prev_block = self.zone1_end or self.fake_free_list_head

            if self.zone2_end is not None:
                while prev_block is not self.zone2_end:
                    curr_block = prev_block.next_free_block

                    if curr_block is None or curr_block is self.fake_free_list_tail:
                        raise RuntimeError("Invalid zone B boundary")

                    if curr_block.num_session_refs > block.num_session_refs:
                        break

                    prev_block = curr_block

            # 只有插入到 B 区末尾时才更新 zone2_end。
            if self.zone2_end is None or prev_block is self.zone2_end:
                self.zone2_end = block

        else:
            logger.debug("append: Appending block id %s to zone A.", block.block_id)

            # A 区保持 FIFO 顺序。
            prev_block = self.zone1_end or self.fake_free_list_head
            self.zone1_end = block

        next_block = prev_block.next_free_block
        if next_block is None:
            raise RuntimeError(f"Invalid insertion position for block {block.block_id}")

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

        only_zone_a_or_empty = self.num_free_blocks == 0 or self.zone1_end is last_block

        blocks_only_zone_a = all(
            not block.is_ephemeral and block.num_session_refs == 0 for block in blocks
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

    def promote_to_zone_a(self, block: KVCacheBlock) -> None:
        """Move an existing free block to the tail of zone A."""
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"promote_to_zone_a() called on invalid block: {block}")

        if block is self.zone1_end:
            return

        # Remove without changing num_free_blocks.
        prev_block = block.prev_free_block
        next_block = block.next_free_block

        if block is self.zone2_end:
            self.zone2_end = (
                prev_block
                if (
                    prev_block is not self.fake_free_list_head
                    and prev_block is not self.zone1_end
                )
                else None
            )

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

        Zone B is ordered by session reference count in ascending order.
        """
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"promote_to_zone_b() called on invalid block: {block}")

        # Remove without changing num_free_blocks.
        old_prev = block.prev_free_block
        old_next = block.next_free_block

        # block 原来是 A 区最后一个节点。
        if block is self.zone1_end:
            self.zone1_end = (
                old_prev if old_prev is not self.fake_free_list_head else None
            )

        # block 原来是 B 区最后一个节点。
        if block is self.zone2_end:
            self.zone2_end = (
                old_prev
                if (
                    old_prev is not self.fake_free_list_head
                    and old_prev is not self.zone1_end
                )
                else None
            )

        old_prev.next_free_block = old_next
        old_next.prev_free_block = old_prev

        # 在 B 区内按照 session 引用数量升序寻找插入位置。
        prev_block = self.zone1_end or self.fake_free_list_head

        if self.zone2_end is not None:
            while prev_block is not self.zone2_end:
                curr_block = prev_block.next_free_block

                if curr_block is None or curr_block is self.fake_free_list_tail:
                    raise RuntimeError("Invalid zone B boundary")

                if curr_block.num_session_refs > block.num_session_refs:
                    break

                prev_block = curr_block

        next_block = prev_block.next_free_block
        if next_block is None:
            raise RuntimeError(
                f"Invalid zone B insertion position for block {block.block_id}"
            )

        prev_block.next_free_block = block
        block.prev_free_block = prev_block

        block.next_free_block = next_block
        next_block.prev_free_block = block

        # B 区为空，或者插入在原 B 区末尾时，更新尾节点。
        if self.zone2_end is None or prev_block is self.zone2_end:
            self.zone2_end = block

    def promote_to_zone_c(self, block: KVCacheBlock) -> None:
        """Move an existing free block into zone C.

        Zone C is ordered by _ttl_expire_at ascending, so blocks
        with a larger TTL are placed closer to the end.
        """
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"promote_to_zone_c() called on invalid block: {block}")

        # 先从原位置摘除，但不修改 num_free_blocks。
        old_prev = block.prev_free_block
        old_next = block.next_free_block

        # block 原来是 A 区最后一个节点。
        if block is self.zone1_end:
            self.zone1_end = (
                old_prev if old_prev is not self.fake_free_list_head else None
            )

        # block 原来是 B 区最后一个节点。
        if block is self.zone2_end:
            self.zone2_end = (
                old_prev
                if (
                    old_prev is not self.fake_free_list_head
                    and old_prev is not self.zone1_end
                )
                else None
            )

        old_prev.next_free_block = old_next
        old_next.prev_free_block = old_prev

        block.prev_free_block = None
        block.next_free_block = None

        # C 区从 A/B 区之后开始，一直到 fake tail。
        prev_block = self.zone2_end or self.zone1_end or self.fake_free_list_head
        curr_block = prev_block.next_free_block

        # 按 _ttl_expire_at 升序寻找插入位置。
        while curr_block is not self.fake_free_list_tail:
            if curr_block is None:
                raise RuntimeError("Invalid zone C boundary")

            # 插入到第一个过期时间更大的 block 前面, 相同过期时间保持 FIFO 顺序
            if curr_block.ttl_expire_at > block.ttl_expire_at:
                break

            prev_block = curr_block
            curr_block = curr_block.next_free_block

        next_block = prev_block.next_free_block
        if next_block is None:
            raise RuntimeError(
                f"Invalid zone C insertion position for block {block.block_id}"
            )

        prev_block.next_free_block = block
        block.prev_free_block = prev_block

        block.next_free_block = next_block
        next_block.prev_free_block = block

    def on_block_meta_changed(self, block: KVCacheBlock) -> None:
        """block metadata（_session_ref_cnt 或 _ttl_expire_at）变化后重新评估分区"""
        if block.ref_cnt > 0 or block.is_null:
            return  # block 被活跃请求使用，不在 free queue 中

        if block.is_ephemeral:
            # C区：ephemeral 保护中，不可分配
            self.promote_to_zone_c(block)
        elif block.num_session_refs > 0 or block.is_offload_block:
            # B区：无 ephemeral 保护但有 session 引用
            self.promote_to_zone_b(block)
        else:
            # A区：无保护、无引用，优先分配
            self.promote_to_zone_a(block)
