# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, G004
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
)
from vllm_ascend.core.agent_hint.params import (
    CacheControlParams,
    ContextManagementEditsParams,
    ContextManagementParams,
    convert_agent_hint_dict,
)
from vllm.v1.request import Request

from vllm_ascend.core.agent_hint.session_event_listener import SessionEventListener



def split_base_block_hashes(
    block_hash: BlockHashWithGroupId,
    block_size: int,
    hash_block_size: int,
) -> list[BlockHash]:
    """split KVCacheBlock block hash to request block hash"""
    assert block_hash is not None
    assert block_size % hash_block_size == 0

    # block.block_hash 是 BlockHashWithGroupId：
    # [拼接后的 BlockHash][4 字节 group_id]
    merged_hash = get_block_hash(block_hash)

    num_base_hashes = block_size // hash_block_size
    assert len(merged_hash) % num_base_hashes == 0

    digest_size = len(merged_hash) // num_base_hashes

    return [BlockHash(merged_hash[i : i + digest_size]) for i in range(0, len(merged_hash), digest_size)]


@dataclass
class SessionBlockRecord:
    """单个 session 对单个 block 的引用记录（仅在 SAM 内部维护）"""

    session_id: str
    block_id: int
    is_ephemeral: bool = False  # 是否受 cache_control ephemeral 保护
    ttl_expire_at: float = 0.0  # ephemeral block 的 TTL 过期时间，0 表示无限制
    created_at: float = 0.0  # 记录创建时间
    block_hash: BlockHashWithGroupId | None = None  # 记录创建时的 block hash，便于调试


@dataclass
class SessionInfo:
    """Session 注册信息"""

    session_id: str
    parent_session_id: str | None = None
    children: set[str] = field(default_factory=set)
    request_ids: set[str] = field(default_factory=set)  # 关联的请求 ID
    created_at: float = 0.0


@dataclass
class EphemeralRange:
    """Ephemeral 保护范围"""

    block_offset: int  # ephemeral 保护的起始 block index
    ttl: float  # TTL 时长（秒）


@dataclass
class EditResponse:
    """session controller针对单个edit返回结构体"""

    session_id: str  # session id
    type: str  # op类型
    op_status: bool  # 执行状态
    expected_op_block_num: int = 0  # 预期编辑block数
    actual_op_block_num: int = 0  # 实际编辑block数
    fail_reason: str = ""  # （可选）执行失败原因


class SessionAwareManager:
    """Session-Aware Manager — 管理 session 生命周期与 block 映射"""

    # TODO：是否已实现分配的时候有ttl才保护
    # TODO：没空间是C区可开放，远程保持访问、保护
    def __init__(self, kv_cache_manager: KVCacheManager):
        self.kv_cache_manager = kv_cache_manager
        self.num_kv_cache_groups = kv_cache_manager.num_kv_cache_groups
        self.block_size = tuple(sprc.block_size for sprc in self.kv_cache_manager.coordinator.single_type_managers)
        self.hash_block_size = self.kv_cache_manager.block_pool.hash_block_size

        # Session注册表(session间的层级关系): session_id → SessionInfo
        self._sessions: dict[str, SessionInfo] = {}
        # session_id → {block_id → SessionBlockRecord}
        self._session_blocks: list[dict[str, dict[int, SessionBlockRecord]]] = [
            {} for _ in range(self.num_kv_cache_groups)
        ]
        # 反向索引：block_id → {session_id → SessionBlockRecord}
        self._block_sessions: list[dict[int, dict[str, SessionBlockRecord]]] = [
            {} for _ in range(self.num_kv_cache_groups)
        ]
        # 只维护当前"请求"的hash表
        self._session_block_hash: dict[str, list[BlockHash]] = {}

        # TTL 管理器（定时轮）
        self._ttl_manager = TTLManager(on_expired=self._on_ttl_expired, notify_func=self._notify_event)

        # Session 控制器
        self._session_controller = SessionController(
            execute_offload=self._execute_offload,
            execute_prefetch=self._execute_prefetch,
            execute_evict=self._execute_evict,
            get_global_block_id_by_session=self._get_session_global_block_ids,
            get_block_hashes_by_session=self._get_session_block_hash,
        )

        # 事件监听器列表
        self._event_listeners: list[SessionEventListener] = []

    def tick(self) -> None:
        self._ttl_manager.tick()

    def on_request_completed(self, request_id: str) -> None:
        self._session_controller.on_request_completed(request_id)

    def on_blocks_allocated_for_request(
        self, request: Request, blocks: KVCacheBlocks, cached_blocks_len_before: tuple[int, ...] | None = None
    ) -> None:
        # 解析引擎层透传的 opaque dict 为 ascend 强类型视图
        hint = convert_agent_hint_dict(request.agent_hint)
        # 如果 request 没有 session_id，则不进行 session 相关的处理
        # 但是清理所有 block 的 session 引用和 TTL，避免残留
        if not hint or hint.session_id is None:
            for group_id, block_ids in enumerate(blocks.get_block_ids()):
                for block_id in block_ids:
                    block = self._get_block(block_id)

                    # block本身清理的session_ref_cnt和ttl_expire_at
                    self._reset_block_state(block)

                    # 清理TTLManager中这个物理 block 的所有旧 session 引用
                    self._clear_old_block_sessions_ttl(block_id)

                    # 清理SAM中这个物理 block 的所有旧 session 引用。
                    self._clear_block_session_refs(block_id)
            return

        is_prefill = request.num_output_tokens == 0
        logger.info(
            f"on_blocks_allocated_for_request, request_id = {request.request_id}, "
            f"session_id = {hint.session_id if hint else None}, "
            f"block_ids = {blocks.get_block_ids()}, request.num_output_tokens: {request.num_output_tokens}"
        )

        self._on_blocks_allocated(
            session_id=hint.session_id,
            parent_session_id=hint.parent_session_id,
            blocks=blocks,
            ephemeral_range=compute_ephemeral_range(hint.cache_control),
            cached_blocks_len_before=cached_blocks_len_before,
            is_prefill=is_prefill,
        )

    def on_block_cache_hit_for_request(self, request: Request, blocks: KVCacheBlocks) -> None:
        hint = convert_agent_hint_dict(request.agent_hint)
        if hint is None or hint.session_id is None:
            return

        logger.info(
            f"on_block_cache_hit_for_request, request_id = {request.request_id}, "
            f"session_id = {hint.session_id if hint else None}, "
            f"block_ids = {blocks.get_block_ids()}"
        )

        self._on_blocks_cache_hit(
            session_id=hint.session_id,
            parent_session_id=hint.parent_session_id,
            blocks=blocks,
            ephemeral_range=compute_ephemeral_range(hint.cache_control),
        )

    def _on_blocks_allocated(
        self,
        session_id: str | None,
        parent_session_id: str | None,
        blocks: KVCacheBlocks,
        ephemeral_range: EphemeralRange | None = None,
        cached_blocks_len_before: tuple[int, ...] | None = None,
        is_prefill: bool = True,
    ) -> None:
        """记录本轮刚刚变为完整状态的 cached blocks。"""

        self._ensure_session_registered(session_id, parent_session_id)

        now = time.monotonic()
        newly_protected_hashes: list[tuple(int, list[BlockHash])] = []
        newly_protected_ttl: float = 0.0
        protected_block_hashes: list[BlockHash] = []

        for group_id, block_ids in enumerate(blocks.get_block_ids()):
            logger.info(f"on_blocks_allocated, block_ids={block_ids}")
            cached_blocks_len = cached_blocks_len_before[group_id] if cached_blocks_len_before else 0

            for ind, block_id in enumerate(block_ids):
                block = self._get_block(block_id)

                # block本身清理的session_ref_cnt和ttl_expire_at
                self._reset_block_state(block)

                # 清理TTLManager中这个物理 block 的所有旧 session 引用
                self._clear_old_block_sessions_ttl(block_id)

                # 清理SAM中这个物理 block 的所有旧 session 引用。
                self._clear_block_session_refs(block_id)

                # allocate_slots理论上只传 newly-cached blocks，保留检查用于防御异常情况。
                if block.block_hash is None:
                    logger.warning("Newly cached block %s has no block hash", block_id)
                    continue

                # SessionBlockRecord所需参数计算
                is_ephemeral = (
                    ephemeral_range is not None
                    and ephemeral_range.ttl > 0
                    and self.block_size[group_id] // self.hash_block_size * (cached_blocks_len + ind)
                    <= ephemeral_range.block_offset
                    and is_prefill
                )

                ttl_expire_at = now + ephemeral_range.ttl if is_ephemeral and ephemeral_range is not None else 0.0

                record = SessionBlockRecord(
                    session_id=session_id,
                    block_id=block_id,
                    is_ephemeral=is_ephemeral,
                    ttl_expire_at=ttl_expire_at,
                    created_at=now,
                    block_hash=block.block_hash,
                )
                self._add_session_block_ref(record, group_id)

                if is_ephemeral:
                    newly_protected_hashes.append(
                        (
                            block_id,
                            split_base_block_hashes(block.block_hash, self.block_size[group_id], self.hash_block_size),
                        )
                    )
                    newly_protected_ttl = ttl_expire_at

                # get_new_blocks，已经把物理 block 的 session_ref_cnt 清零。
                # 使用真实 metadata 做校准，可以兼容异常重复回调：
                current_ref_count = block.num_session_refs
                self._update_block_meta(
                    block_id,
                    delta_ref=1 - current_ref_count,
                    ttl_expire_at=ttl_expire_at,
                )

        if ephemeral_range:
            scale_factor = max(self.block_size) // self.hash_block_size
            session_block_hash = self._session_block_hash.get(session_id)
            block_end = (ephemeral_range.block_offset + scale_factor) // scale_factor * scale_factor
            if ephemeral_range.block_offset + 1 > len(session_block_hash):
                logger.warning(f"ephemeral range exceed session block hash. session id:{session_id}")
                block_end = len(session_block_hash)

            if newly_protected_hashes and session_block_hash:
                protected_block_hashes = session_block_hash[0:block_end]

        self._ttl_manager.register(
            block_infos=newly_protected_hashes,
            session_id=session_id,
            expire_at=newly_protected_ttl,
            protected_block_hashes=protected_block_hashes,
        )

    def _on_blocks_cache_hit(
        self,
        session_id: str | None,
        parent_session_id: str | None,
        blocks: KVCacheBlocks,
        ephemeral_range: EphemeralRange | None = None,
    ) -> None:
        """prefix cache 命中时通知 SAM"""

        self._ensure_session_registered(session_id, parent_session_id)

        # 最新request的ttl时间
        now = time.monotonic()
        newly_protected_hashes: list[tuple(int, list[BlockHash])] = []
        newly_protected_ttl: float = 0.0
        protected_block_hashes: list[BlockHash] = []

        for group_id, block_ids in enumerate(blocks.get_block_ids()):
            for ind, block_id in enumerate(block_ids):
                # cache hit 的 block 应当已经完整且具有 hash。
                block = self._get_block(block_id)
                if block.block_hash is None:
                    logger.warning("Cache-hit block %s has no block hash; ", block_id)
                    break

                # 命中后SessionBlockRecord要刷新的参数
                is_ephemeral = (
                    ephemeral_range is not None
                    and ephemeral_range.ttl > 0
                    and self.block_size[group_id] // self.hash_block_size * ind <= ephemeral_range.block_offset
                )
                requested_expire_at = now + ephemeral_range.ttl if is_ephemeral and ephemeral_range is not None else 0.0

                # 检查 session 是否已注册对 block 的引用
                is_registered = self._is_session_block_registered(session_id, block_id, group_id)

                # 如果已注册，刷新 TTL
                if is_registered:
                    record = self._session_blocks[group_id][session_id][block_id]

                    # 刷新TTL, session引用中的刷新为本次的；block本身的刷新为最长的那个
                    if is_ephemeral:
                        new_expire_at = max(record.ttl_expire_at, requested_expire_at)

                        record.is_ephemeral = True
                        record.ttl_expire_at = new_expire_at

                        # 刷新双向索引，确保两边引用同一个最新 record。
                        self._add_session_block_ref(record, group_id)

                        # self._ttl_manager.update(block_id, session_id, new_expire_at)
                        newly_protected_hashes.append(
                            (
                                block_id,
                                split_base_block_hashes(
                                    block.block_hash, self.block_size[group_id], self.hash_block_size
                                ),
                            )
                        )
                        newly_protected_ttl = new_expire_at

                        # 一个 block 可能被多个 session 引用，block metadata 应使用
                        # 所有 session 记录中最晚的过期时间。
                        block_expire_at = max(
                            item.ttl_expire_at for item in self._block_sessions[group_id][block_id].values()
                        )
                        self._update_block_meta(
                            block_id,
                            ttl_expire_at=block_expire_at,
                        )
                # 未注册则创建新的引用记录
                else:
                    record = SessionBlockRecord(
                        session_id=session_id,
                        block_id=block_id,
                        is_ephemeral=is_ephemeral,
                        ttl_expire_at=requested_expire_at,
                        created_at=now,
                        block_hash=block.block_hash,
                    )
                    self._add_session_block_ref(record, group_id)

                    if is_ephemeral:
                        newly_protected_hashes.append(
                            (
                                block_id,
                                split_base_block_hashes(
                                    block.block_hash, self.block_size[group_id], self.hash_block_size
                                ),
                            )
                        )
                        newly_protected_ttl = requested_expire_at

                        block_expire_at = max(
                            item.ttl_expire_at for item in self._block_sessions[group_id][block_id].values()
                        )
                    else:
                        # None 表示不覆盖其他 session 已经设置的 block TTL。
                        block_expire_at = None

                    self._update_block_meta(
                        block_id,
                        delta_ref=+1,
                        ttl_expire_at=block_expire_at,
                    )
        if ephemeral_range:
            scale_factor = max(self.block_size) // self.hash_block_size
            session_block_hash = self._session_block_hash.get(session_id)
            block_end = (ephemeral_range.block_offset + scale_factor) // scale_factor * scale_factor
            if ephemeral_range.block_offset + 1 > len(session_block_hash):
                logger.warning(f"ephemeral range exceed session block hash. session id:{session_id}")
                block_end = len(session_block_hash)

            if newly_protected_hashes and session_block_hash:
                protected_block_hashes = session_block_hash[0:block_end]

        self._ttl_manager.register(
            newly_protected_hashes, session_id, newly_protected_ttl, protected_block_hashes=protected_block_hashes
        )

    def _on_ttl_expired(self, block_id: int, session_id: str) -> None:
        """ephemeral block TTL 到期回调"""
        logger.info(f"working on _on_ttl_expired in SAM for block {block_id} and session id {session_id}")

        group_id = -1
        for group_idx in range(self.num_kv_cache_groups):
            if block_id in self._block_sessions[group_idx]:
                group_id = group_idx
                break

        if group_id == -1:
            logger.warning(f"can not find block_id: {block_id} in all groups")
            return

        record = self._block_sessions[group_id].get(block_id, {}).get(session_id)
        if record is None or not record.is_ephemeral:
            return

        # 移除 SAM 内部记录
        self._remove_session_block_ref(session_id, block_id, group_id)

        # 统一接口：清除 TTL + 减少 session 引用
        # 这会导致 block 的 is_ephemeral() 返回 False
        # 且 _session_ref_cnt 减 1
        # FreeKVCacheBlockQueue.on_block_meta_changed 自动重评估分区
        self._update_block_meta(
            block_id,
            delta_ref=-1,
            ttl_expire_at=0.0,
        )

    def _ensure_session_registered(self, session_id: str, parent_session_id: str) -> None:
        """确保 session 已注册（SAM 内部）"""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionInfo(
                session_id=session_id,
                parent_session_id=parent_session_id,
                created_at=time.monotonic(),
            )
            # 如果有父 session，更新父 session 的 children, children不能是自己
            if parent_session_id and parent_session_id in self._sessions and parent_session_id != session_id:
                self._sessions[parent_session_id].children.add(session_id)

    def _clear_block_session_refs(self, block_id: int) -> None:
        """清除指定 block 的所有 session 引用（SAM 内部）"""
        for group_id in range(len(self._block_sessions)):
            if block_id in self._block_sessions[group_id]:
                for session_id in list(self._block_sessions[group_id][block_id].keys()):
                    self._remove_session_block_ref(session_id, block_id, group_id)

    def _add_session_block_ref(self, record: SessionBlockRecord, group_id: int) -> None:
        """添加 session 对 block 的引用（SAM 内部）"""
        self._session_blocks[group_id].setdefault(record.session_id, {})[record.block_id] = record
        self._block_sessions[group_id].setdefault(record.block_id, {})[record.session_id] = record

    def _is_session_block_registered(self, session_id: str, block_id: int, group_id: int) -> bool:
        """检查 session 是否已注册对 block 的引用（SAM 内部）"""
        return session_id in self._session_blocks[group_id] and block_id in self._session_blocks[group_id][session_id]

    def _remove_session_block_ref(self, session_id: str, block_id: int, group_id: int) -> None:
        """移除 session 对 block 的引用（SAM 内部）"""
        if session_id in self._session_blocks[group_id]:
            self._session_blocks[group_id][session_id].pop(block_id, None)
            if not self._session_blocks[group_id][session_id]:
                del self._session_blocks[group_id][session_id]
        if block_id in self._block_sessions[group_id]:
            self._block_sessions[group_id][block_id].pop(session_id, None)
            if not self._block_sessions[group_id][block_id]:
                del self._block_sessions[group_id][block_id]

    def _free_session(self, session_id: str) -> list:
        """清理指定 session 的所有 block 引用"""
        block_hashes = []

        for group_id in range(self.num_kv_cache_groups - 1, -1, -1):
            block_ids = []
            if session_id in self._session_blocks[group_id]:
                for block_id in list(self._session_blocks[group_id][session_id].keys())[::-1]:
                    cur_block_session = self._block_sessions[group_id].get(block_id, {})
                    record = cur_block_session.get(session_id)
                    if record and record.is_ephemeral:
                        self._ttl_manager.remove(block_id, session_id)

                    self._remove_session_block_ref(session_id, block_id, group_id)

                    remaining_records = self._block_sessions[group_id].get(block_id, {}).values()
                    latest_ttl_expire_at = max(
                        (remaining_record.ttl_expire_at for remaining_record in remaining_records),
                        default=0.0,
                    )

                    self._update_block_meta(block_id, delta_ref=-1, ttl_expire_at=latest_ttl_expire_at)

                    block_ids.append(block_id)
            logger.info(f"free: session id:{session_id}, group_id:{group_id}, block id:{block_ids}")

        if session_id in self._session_block_hash:
            block_hashes = self._session_block_hash[session_id]
            del self._session_block_hash[session_id]

        # 移除 session 注册信息
        if session_id in self._sessions:
            parent_session_id = self._sessions[session_id].parent_session_id
            if parent_session_id and parent_session_id in self._sessions:
                self._sessions[parent_session_id].children.discard(session_id)
            del self._sessions[session_id]

        return block_hashes

    def _free_session_tree(self, session_id: str) -> list:
        """递归清理session及其所有子session"""
        session_info = self._sessions.get(session_id)
        if session_info is None:
            return []

        block_hashes_all = []
        for child_sid in list(self._sessions[session_id].children):
            block_hashes_all.extend(self._free_session_tree(child_sid))

        block_hashes_all.extend(self._free_session(session_id))

        logger.debug(
            f"Free session tree for session {session_id}: "
            f"current session_to_blocks: {self._session_blocks}, "
            f"session_info: {self._sessions}, "
        )

        return block_hashes_all

    def _execute_offload(
        self,
        session_id: str,
        block_start: int,
        block_end: int,
        block_ids_all_group: tuple[list[int], ...],
        is_session: bool,
    ) -> int:
        """卸载指定范围的 block — 减少 session 引用 + 清除当前session的TTL（通知TTLManager）"""
        affected_block_hashes: list[BlockHash] = []

        for group_id in range(self.num_kv_cache_groups - 1, -1, -1):
            assert self.block_size[group_id] % self.hash_block_size == 0, (
                f"block_size {self.block_size[group_id]} is not a multiple of {self.hash_block_size}"
            )

            block_ids = block_ids_all_group[group_id]

            for block_id in block_ids[::-1]:
                cur_block_session = self._block_sessions[group_id].get(block_id, {})
                record = cur_block_session.get(session_id)
                if len(cur_block_session) == 0 or record is None:
                    continue

                if record.is_ephemeral:
                    self._ttl_manager.remove(block_id=block_id, session_id=session_id, is_notify_spm=False)

                    # affected_block_hashes.extend(
                    #     split_base_block_hashes(record.block_hash, self.block_size[group_id], self.hash_block_size))

                self._remove_session_block_ref(session_id, block_id, group_id)

                remaining_records = self._block_sessions[group_id].get(block_id, {}).values()
                latest_ttl_expire_at = max(
                    (remaining_record.ttl_expire_at for remaining_record in remaining_records),
                    default=0.0,
                )
                self._update_block_meta(
                    block_id,
                    delta_ref=-1,
                    ttl_expire_at=latest_ttl_expire_at,
                    is_offload_block=True,
                )

        block_start_e, block_end_e = self._expand_session_block_range(session_id, block_start, block_end)
        affected_block_hashes.extend(self._session_block_hash.get(session_id, [])[block_start_e:block_end_e])
        res = len(affected_block_hashes)

        self._ttl_manager.register(
            block_infos=[],
            session_id=session_id,
            expire_at=time.monotonic() + 3600,
            protected_block_hashes=affected_block_hashes,
        )

        return res

    def _execute_prefetch(self, session_id: str, block_hashes: list[BlockHash], is_session: bool = False) -> int:
        """通知 SPM 创建远端预取任务。"""
        # logger.info(f"block_hashes {block_hashes}")
        self._notify_event(
            "context_management_prefetch",
            session_id=session_id,
            block_hashes=block_hashes,
        )
        # TODO: 后续返回当前session及其子session的block hash
        return len(block_hashes)

    def _execute_evict(
        self,
        session_id: str,
        block_start: int,
        block_end: int,
        block_ids_all_group: tuple[list[int], ...],
        is_session: bool,
    ) -> int:
        """驱逐指定范围的 block — 减少引用 + 清除当前session的TTL
        清除本地引用，并通知 SPM 停止对应远端 PoolKey 的 Keep-Alive
        """
        res = 0
        if not is_session:
            affected_block_hashes: list[BlockHash] = []

            for group_id in range(self.num_kv_cache_groups - 1, -1, -1):
                assert self.block_size[group_id] % self.hash_block_size == 0, (
                    f"block_size {self.block_size[group_id]} is not a multiple of {self.hash_block_size}"
                )

                block_ids = block_ids_all_group[group_id]

                for block_id in block_ids[::-1]:
                    cur_block_session = self._block_sessions[group_id].get(block_id, {})
                    record = cur_block_session.get(session_id)
                    if len(cur_block_session) == 0 or record is None:
                        continue

                    # affected_block_hashes.extend(
                    #     split_base_block_hashes(record.block_hash, self.block_size[group_id], self.hash_block_size))

                    self._remove_session_block_ref(session_id, block_id, group_id)

                    remaining_records = self._block_sessions[group_id].get(block_id, {}).values()
                    latest_ttl_expire_at = max(
                        (remaining_record.ttl_expire_at for remaining_record in remaining_records),
                        default=0.0,
                    )
                    self._update_block_meta(
                        block_id,
                        delta_ref=-1,
                        ttl_expire_at=latest_ttl_expire_at,
                    )

            block_start_e, block_end_e = self._expand_session_block_range(session_id, block_start, block_end)
            affected_block_hashes.extend(self._session_block_hash.get(session_id, [])[block_start_e:block_end_e])

        else:
            affected_block_hashes = self._free_session_tree(session_id)

        self._ttl_manager.remove(block_id=-1, session_id=session_id)
        res = len(affected_block_hashes)
        return res

    def _get_session_global_block_ids(
        self,
        session_id: str,
        block_start: int,
        block_end: int,
    ) -> tuple[list[int], ...]:
        """将基础 hash block 范围映射为各 KV cache group 的物理 block id。

        当某个 KV cache group 的 block size 大于 hash_block_size 时，只要
        物理 block 与指定逻辑范围有重叠，就返回该物理 block。
        """
        result: tuple[list[int], ...] = tuple([] for _ in range(self.num_kv_cache_groups))

        base_block_hashes = self._session_block_hash.get(session_id)
        if not base_block_hashes:
            logger.warning(f"There is no hash for the session_id:{session_id} in the _session_block_hash.")
            return result

        base_block_count = len(base_block_hashes)

        if block_end == -1:
            block_end = base_block_count

        # 防止负数下标被 Python 当成从末尾索引。
        if block_start < 0 or block_end < 0:
            logger.warning(
                "Invalid block range for session %s: [%s, %s)",
                session_id,
                block_start,
                block_end,
            )
            return result

        block_end = min(block_end, base_block_count)

        if block_start >= block_end:
            return result

        for group_id in range(self.num_kv_cache_groups):
            # 获取该 session_id 第 group_id 的 block_ids(即blocks)
            session_blocks = self._session_blocks[group_id].get(session_id)
            # 该group的block_size可能比较大，还没满，继续下一个group
            if not session_blocks:
                continue

            group_block_size = self.block_size[group_id]
            assert group_block_size % self.hash_block_size == 0, (
                f"block_size {group_block_size} is not a multiple of {self.hash_block_size}"
            )

            # 将基础 block hash 转换成对应block size的BlockHash
            scale_factor = group_block_size // self.hash_block_size
            if scale_factor == 1:
                group_block_hashes = base_block_hashes
            else:
                group_block_hashes = BlockHashListWithBlockSize(
                    base_block_hashes,
                    self.hash_block_size,
                    group_block_size,
                )

            # 逻辑范围 [block_start, block_end) 与物理 block 有重叠即选中。
            group_block_start = block_start // scale_factor
            group_block_end = (block_end + scale_factor - 1) // scale_factor
            group_block_end = min(group_block_end, len(group_block_hashes))

            if group_block_start >= group_block_end:
                continue

            # 每个 group 建立 [hash: block_id] 映射
            hash_block: dict[BlockHash, int] = {}

            for block_id in session_blocks:
                block_hash_with_group_id = session_blocks[block_id].block_hash

                # 未填满或尚未进入 prefix cache 的 block 可能没有 hash。
                if block_hash_with_group_id is None:
                    logger.debug(
                        "Skip unhashed block %s for session %s, group %s",
                        block_id,
                        session_id,
                        group_id,
                    )
                    continue

                physical_hash = get_block_hash(block_hash_with_group_id)

                # 理论上同一 group 的相同完整 hash 应映射到同一缓存内容。
                # 保留第一次记录，避免覆盖造成结果不稳定。
                hash_block.setdefault(physical_hash, block_id)

            for group_index in range(group_block_start, group_block_end):
                target_hash = group_block_hashes[group_index]
                block_id = hash_block.get(target_hash)

                if block_id is None:
                    logger.debug(
                        f"block not found, target block hash:{target_hash}, block index:{group_index * scale_factor}"
                    )
                    continue

                result[group_id].append(block_id)

        return result

    def _get_session_block_hash(self, session_id: str, block_start: int, block_end: int) -> list[BlockHash]:
        if session_id in self._session_block_hash:
            block_hash_len = len(self._session_block_hash[session_id])
            assert block_start >= 0 and block_end <= block_hash_len
            if block_end == -1:
                block_end = block_hash_len
            return self._session_block_hash[session_id][block_start:block_end]
        else:
            return []

    def add_event_listener(self, listener: SessionEventListener):
        """注册事件监听器（SPM 调用）"""
        self._event_listeners.append(listener)

    def _notify_event(self, event_type: str, **kwargs):
        """通知所有监听器"""
        ret = 0
        for listener in tuple(self._event_listeners):
            handler = getattr(listener, f"on_{event_type}", None)
            if handler is None:
                continue

            try:
                ret = handler(**kwargs)
            except Exception:
                logger.exception(
                    "Session event listener %r failed while handling %s",
                    listener,
                    event_type,
                )
        return ret

    def register_agent_hint(
        self,
        request_id: str,
        session_id: str | None,
        context_management: ContextManagementParams | None,
    ) -> list[Any] | None:
        """register context management to session controller"""
        logger.info(f"register context management with req id {request_id} session id {session_id}")
        return self._session_controller.process_request_edits(request_id, session_id, context_management)

    def register_session_block_hash(self, session_id: str, block_hashes: list[BlockHash]) -> None:
        """register session block hash to session aware manager"""
        self._session_block_hash[session_id] = block_hashes

    def _get_block(self, block_id: int) -> KVCacheBlock:
        return self.kv_cache_manager.block_pool.blocks[block_id]

    def _reset_block_state(self, block: KVCacheBlock) -> None:
        """
        Reset the session state of a block, clearing its session
        reference count and TTL expiration time.
        """
        block.reset_session_state()

    def _update_block_meta(
        self,
        block_id: int,
        delta_ref: int = 0,
        ttl_expire_at: float | None = None,
        is_offload_block: bool | None = None,
    ) -> None:
        """Update Agent Hint metadata and re-evaluate the free queue zone."""
        block_pool = self.kv_cache_manager.block_pool
        block = block_pool.blocks[block_id]
        session_ref_cnt = (
            max(0, block.num_session_refs + delta_ref) if delta_ref != 0 else None
        )
        block.set_session_state(
            session_ref_cnt=session_ref_cnt,
            ttl_expire_at=ttl_expire_at,
            is_offload_block=is_offload_block,
        )
        if block.ref_cnt == 0 and not block.is_null:
            block_pool.free_block_queue.on_block_meta_changed(block)

    def _expand_session_block_range(
        self,
        session_id: str,
        block_start: int,
        block_end: int,
    ) -> tuple[int, int]:
        """扩展左闭右开区间，使其包含所有 group 涉及的完整 block。

        输入和输出均为 hash_block_size 粒度，区间语义均为
        [block_start, block_end)。
        """
        assert block_start >= 0
        assert block_end >= block_start

        # 空区间没有涉及任何物理 block。
        if block_start == block_end:
            return block_start, block_end

        base_block_hashes = self._session_block_hash.get(session_id)
        if not base_block_hashes:
            logger.warning(f"There is no hash for the session_id:{session_id} in the _session_block_hash.")
            return 0, 0

        base_block_count = len(base_block_hashes)

        expanded_start = block_start
        expanded_end = block_end

        for group_block_size in self.block_size:
            assert group_block_size % self.hash_block_size == 0

            scale_factor = group_block_size // self.hash_block_size

            # 包含 block_start 的物理 block 起点。
            group_block_start = (block_start // scale_factor) * scale_factor

            # 包含 block_end 前一个位置的物理 block 的右边界。
            group_block_end = (block_end + scale_factor - 1) // scale_factor * scale_factor

            expanded_start = min(expanded_start, group_block_start)
            expanded_end = max(expanded_end, group_block_end)

        expanded_end = min(expanded_end, base_block_count)
        return expanded_start, expanded_end

    def _clear_old_block_sessions_ttl(self, block_id: int) -> None:
        # 清理TTLManager中这个物理 block 的所有旧 session 引用
        for group_id in range(len(self._block_sessions)):
            old_session_ids = list(self._block_sessions[group_id].get(block_id, {}).keys())
            for old_session_id in old_session_ids:
                record = self._block_sessions[group_id][block_id].get(old_session_id)
                if record and record.is_ephemeral:
                    self._ttl_manager.remove(block_id, old_session_id)


@dataclass
class TTLBlockEntry:
    block_id: int
    session_id: str
    ttl_expire_at: float
    block_hashes: list[BlockHash]
    update_SPM_only: bool = False


class TTLTimerWheel:
    """Best-effort timer wheel for TTL-protected free KV cache blocks.

    This class does not mutate the free-block queue. It only tracks when a
    block may become eligible for promotion out of zone C. Callers should
    validate that returned blocks are still in the free queue, then call
    FreeKVCacheBlockQueue.promote_to_zone_a/b() as appropriate.
    """

    def __init__(self, tick_count: int = 60):
        """Initialize the timer wheel.

        Parameters
        ----------
        tick_count : int
            Number of slots in the wheel. Each slot corresponds to one
            time unit (second). A block whose TTL expires at time T is
            placed in slot ``int(T) % tick_count``. The wheel wraps
            around, so tick_count also determines the maximum TTL
            range that can be uniquely tracked.
        """
        self.slots: list[list[TTLBlockEntry]] = [[] for _ in range(tick_count)]
        self.current_slot = 0
        self.tick_count = tick_count

        logger.info(f"Initialized TTLTimerWheel with {tick_count} slots.")

    def insert(self, block: TTLBlockEntry, expire_at: float) -> None:
        """Insert a block into the wheel at the slot corresponding to
        its expiration time.

        Parameters
        ----------
        block : TTLBlockEntry
            The block entry to track.
        expire_at : float
            Absolute timestamp (e.g. time.monotonic()) at which the
            block's TTL expires. The block is placed in slot
            ``int(expire_at) % tick_count``.
        """
        tick_index = int(expire_at) % self.tick_count
        self.slots[tick_index].append(block)

        logger.debug(
            f"Inserting block {block.block_id} into TTLTimerWheel, "
            f"block.session_id={block.session_id},"
            f"block.ttl_expire_at={block.ttl_expire_at:.2f}, "
        )

    def remove(self, block: TTLBlockEntry) -> None:
        """Remove a block from the wheel.

        Uses the block's ``ttl_expire_at`` to locate its slot.
        If the block has already been collected by ``advance()``
        (i.e. its TTL already expired), it won't be found in the
        wheel and a warning is logged instead of raising an error.

        Parameters
        ----------
        block : TTLBlockEntry
            The block entry to remove. Must be the same object that
            was passed to ``insert()`` (uses list.remove identity
            check).
        """
        tick_index = int(block.ttl_expire_at) % self.tick_count
        slot = self.slots[tick_index]
        try:
            slot.remove(block)
        except ValueError:
            logger.warning(
                "Block %d not found in slot %d (expire_at=%.2f), may have already expired.",
                block.block_id,
                tick_index,
                block.ttl_expire_at,
            )
            return

        logger.debug(
            "Removed block %d from TTLTimerWheel slot %d, session_id=%s, expire_at=%.2f.",
            block.block_id,
            tick_index,
            block.session_id,
            block.ttl_expire_at,
        )

    def advance(self, now: float) -> list[TTLBlockEntry]:
        """Advance the wheel to the current time and return all blocks
        whose TTL has expired.

        Moves ``current_slot`` forward to ``int(now) % tick_count``,
        collecting and clearing every slot along the way. The returned
        blocks are those whose TTL deadline falls in the time range
        between the previous position and the current time.

        Important: this method only **identifies** expired blocks; it
        does not mutate the free-block queue or change block zones.
        Callers should validate that returned blocks are still in the
        free queue, then apply the appropriate zone transition.

        Parameters
        ----------
        now : float
            Current time, typically ``time.monotonic()``.

        Returns
        -------
        list[TTLBlockEntry]
            All blocks that expired between the previous wheel
            position and ``now``.
        """
        expired = []
        target_slot = int(now) % self.tick_count
        while self.current_slot != target_slot:
            expired.extend(self.slots[self.current_slot])
            self.slots[self.current_slot].clear()
            self.current_slot = (self.current_slot + 1) % self.tick_count
        if len(expired) > 0:
            logger.debug(
                f"Advancing TTLTimerWheel to time {now:.2f}, "
                f"(current_slot={self.current_slot})."
                f" Expired blocks: {[block.block_id for block in expired]}."
            )

        return expired


class TTLManager:
    """Session block 的 TTL 管理器。

    为 session 中受 TTL 保护的 block 提供过期跟踪。每个 block 以
    ``(block_id, session_id)`` 为键，绑定一个绝对过期时间
    ``ttl_expire_at``。内部使用 TTLTimerWheel 做 best-effort 的
    过期发现，配合 ``_entries`` 字典做精确校验。

    核心流程：
      1. **register** — 将 block 注册到 timer wheel，同时通知 SPM
         该 block 处于 TTL 保护状态（"session_blocks_protected"）。
         若同一 key 已存在且新过期时间更晚，则更新；否则保留旧值。
      2. **tick** — 推进 timer wheel，收集已过期 entry。timer wheel
         按 slot 收集，同一 slot 内可能包含尚未真正过期的 entry
         （best-effort），因此 ``tick`` 会二次校验
         ``now >= entry.ttl_expire_at``，仅对确认过期的 entry 执行
         清理并回调 ``on_expired(block_id, session_id)``。
      3. **remove** — 显式移除某个 block 的 TTL 跟踪（如 session
         被主动释放），同时通知 SPM TTL 已失效
         （"session_ttl_expired"）。

    与 SPM 的交互：
      - register 时通过 ``_spm_notify_func("session_blocks_protected", ...)``
        通知 SPM block 处于 TTL 保护，避免 SPM 过早回收。
      - 过期或 remove 时通过 ``_spm_notify_func("session_ttl_expired", ...)``
        通知 SPM TTL 已失效，SPM 可据此决定是否回收该 session 的 block。
    """

    def __init__(
        self,
        on_expired: Callable[[int, str], None],
        notify_func: Callable[..., None],
    ):
        """
        Parameters
        ----------
        on_expired : Callable[[int, str], None]
            Block 过期时的回调，签名为 ``on_expired(block_id, session_id)``。
            调用方在此回调里执行实际的 block 释放 / zone 转换。
        notify_func : Callable[..., None]
            SPM 通知函数，用于向 SessionAwarePoolingManager 发送事件。
            register 时发送 "session_blocks_protected"，过期/remove 时
            发送 "session_ttl_expired"，使 SPM 感知 block 的 TTL 状态变化。
        """
        self._on_expired = on_expired
        self._spm_notify_func = notify_func
        self._entries: dict[tuple[int, str], TTLBlockEntry] = {}
        self._timer_wheel = TTLTimerWheel(tick_count=3600)

        logger.info("TTLManager initialized with on_expired=%s", getattr(on_expired, "__name__", repr(on_expired)))

    def register(
        self,
        block_infos: list[tuple(int, list[BlockHash])],
        session_id: str,
        expire_at: float,
        protected_block_hashes: list[BlockHash] = None,
    ) -> None:
        """注册或更新一个 block 的 TTL。

        如果 ``(block_id, session_id)`` 已存在：
          - 当新的 ``expire_at`` 比旧的更晚时，先从 timer wheel 移除旧
            entry，更新 expire_at 后重新插入；
          - 否则忽略（保留更晚的过期时间）。
        如果不存在：创建新的 TTLBlockEntry 并插入 timer wheel。
        """
        protected_hash_len = 0
        if protected_block_hashes is not None:
            block_infos.append((0, protected_block_hashes))
            protected_hash_len = len(protected_block_hashes)

        logger.debug(f"TTL Manager: working to register {len(block_infos)} blocks into timer wheel")
        for block_info in block_infos:
            key = (block_info[0], session_id)
            if key in self._entries:
                old_entry = self._entries[key]
                if expire_at > old_entry.ttl_expire_at:
                    self._timer_wheel.remove(old_entry)
                    old_entry.ttl_expire_at = expire_at
                    old_entry.block_hashes = block_info[1]
                    self._timer_wheel.insert(old_entry, expire_at)

            else:
                entry = TTLBlockEntry(
                    block_id=block_info[0], session_id=session_id, ttl_expire_at=expire_at, block_hashes=block_info[1]
                )
                self._entries[key] = entry
                self._timer_wheel.insert(entry, expire_at)
        logger.info(
            f"Register session id {session_id} with ttl {expire_at} and protected hash length {protected_hash_len} in TTL Manager"
        )

        protected_key = (0, session_id)
        if protected_key in self._entries:
            entry = self._entries[protected_key]
            self._spm_notify_func("session_blocks_protected", session_id=session_id, block_hashes=entry.block_hashes)

    def remove(self, block_id: int, session_id: str, is_notify_spm: bool = True) -> None:
        """显式移除一个 block 的 TTL 跟踪。

        同时从 ``_entries`` 和 timer wheel 中删除。如果不存在则静默忽略。
        """
        block_id = block_id if block_id > 0 else 0
        key = (block_id, session_id)
        if key in self._entries:
            entry = self._entries.pop(key)
            self._timer_wheel.remove(entry)
            if block_id == 0:
                remove_hashes = set()
                remove_hashes.update(entry.block_hashes)
                self._spm_notify_func("session_ttl_expired", session_id=session_id, block_hashes=list(remove_hashes))
        else:
            logger.warning(f"Could not find block_id {block_id} and session id {session_id} in TTL Manager")

    def tick(self, now: float | None = None) -> None:
        """推进 timer wheel，处理所有已过期的 entry。

        对每个由 timer wheel 收集到的过期 entry，先校验 ``now >= expire_at``
        （timer wheel 是 best-effort，slot 里可能含尚未真正过期的 entry），
        然后从 ``_entries`` 移除并调用 ``on_expired`` 回调。

        Parameters
        ----------
        now : float, optional
            当前时间戳，默认 ``time.monotonic()``。
        """
        now = now if now is not None else time.monotonic()
        expired_entries = self._timer_wheel.advance(now)
        for entry in expired_entries:
            if now >= entry.ttl_expire_at:
                self._entries.pop((entry.block_id, entry.session_id), None)
                if entry.block_id == 0:
                    self._spm_notify_func(
                        "session_ttl_expired", session_id=entry.session_id, block_hashes=entry.block_hashes
                    )
                else:
                    self._on_expired(entry.block_id, entry.session_id)


def example_expired_callback(block_id: int, session_id: str) -> None:
    logger.info(f"block_id {block_id} and session_id {session_id} is processing on TTL expiration")


class SessionController:
    # TODO: 考虑预取请求有content / session已被清理，重新计算hash
    """Context management edits 执行器。

    处理请求携带的 context_management.edits，根据
    manage_request 标志决定是立即执行还是在请求完成后执行。
    通过注册回调执行实际的 block 操作，不感知具体的
    offload/prefetch/evict 实现。

    Required callbacks
    ------------------
    execute_offload  (block_ids: list[int], session_id: str) -> None
        执行 offload 操作。
    execute_prefetch (block_ids: list[int], session_id: str) -> None
        执行 prefetch 操作。
    execute_evict   (block_ids: list[int], session_id: str) -> None
        执行 evict 操作。
    """

    _REQUIRED_CALLBACKS: list[str] = [
        "execute_offload",
        "execute_prefetch",
        "execute_evict",
        "get_global_block_id_by_session",
        "get_block_hashes_by_session",
    ]

    def __init__(
        self,
        callbacks: dict[str, Callable[..., Any]] | None = None,
        **kw_callbacks: Callable[..., Any],
    ):
        """
        Parameters
        ----------
        callbacks : dict[str, Callable], optional
            A mapping of {name: fn} to register in bulk.
        **kw_callbacks
            Same as *callbacks* but passed as keyword arguments.
        """
        self._registry: dict[str, Callable[..., Any]] = {}

        if callbacks:
            self._registry.update(callbacks)
        if kw_callbacks:
            self._registry.update(kw_callbacks)

        missing = [name for name in self._REQUIRED_CALLBACKS if name not in self._registry]
        if missing:
            raise ValueError(
                f"Missing required callbacks: {missing}. Provide them via `callbacks` dict or keyword arguments."
            )

        # request_id → [(edit, session_id), ...]
        self._pending_edits: dict[str, list[tuple[ContextManagementEditsParams, str]]] = {}

        logger.info("SessionController initialized with callbacks: %s", list(self._registry))

    # ------------------------------------------------------------------
    #  Registry management
    # ------------------------------------------------------------------

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register (or overwrite) a callback by *name*."""
        self._registry[name] = fn
        logger.debug("Registered callback: %s", name)

    def unregister(self, name: str) -> None:
        """Remove a callback.  Required callbacks cannot be removed."""
        if name in self._REQUIRED_CALLBACKS:
            raise ValueError(f"Cannot unregister required callback '{name}'.")
        self._registry.pop(name, None)
        logger.debug("Unregistered callback: %s", name)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def process_request_edits(
        self,
        request_id: str,
        session_id: str | None,
        context_management: ContextManagementParams | None,
    ) -> list[EditResponse] | None:
        """处理请求携带的 context_management edits。

        Parameters
        ----------
        request_id : str
            请求唯一标识。
        session_id : str | None
            请求所属的 session。为 None 时直接忽略 edits。
        context_management : ContextManagementParams | None
            请求中的 context_management 字段。
        """
        if context_management is None or context_management.edits is None:
            return

        if session_id is None:
            logger.warning("request %s has context_management.edits but no session_id, skipping.", request_id)
            return

        if context_management.manage_request:
            # 管理请求：不执行请求本身，直接执行所有 edits
            logger.info(
                "Processing manage_request edits for request %s, session %s, %d edits.",
                request_id,
                session_id,
                len(context_management.edits),
            )
            edit_results = []
            for edit in context_management.edits:
                edit_results.append(self._execute_single_edit(edit, session_id))
                if edit_results[-1].op_status:
                    logger.info(
                        f"Context management result: session {session_id} type {edit.type} "
                        f"status {edit_results[-1].op_status} "
                        f"block start {edit.block_start} block end (not included) {edit.block_end} "
                        f"actual_op_block_num {edit_results[-1].actual_op_block_num}"
                    )
                else:
                    logger.info(
                        f"Context management result: session {session_id} type {edit.type} "
                        f"status {edit_results[-1].op_status} "
                        f"block start {edit.block_start} block end (not included) {edit.block_end} "
                        f"reason {edit_results[-1].fail_reason}"
                    )
            return edit_results
        else:
            # 普通请求：记录 edits，在请求完成后执行
            logger.info(
                "Deferring %d edits for request %s, session %s. Prefetch will not be performed after request finish",
                len(context_management.edits),
                request_id,
                session_id,
            )
            self._pending_edits[request_id] = [
                (edit, session_id) for edit in context_management.edits if edit.type != "prefetch"
            ]

    def on_request_completed(self, request_id: str) -> None:
        """请求完成后执行其挂起的 edits。"""
        pending = self._pending_edits.pop(request_id, None)
        if pending is None:
            return

        logger.info("Executing %d deferred edits for completed request %s.", len(pending), request_id)
        for edit, session_id in pending:
            result = self._execute_single_edit(edit, session_id)
            if result.op_status:
                logger.info(
                    f"Context management result: session {session_id} type {edit.type} "
                    f"status {result.op_status} "
                    f"block start {edit.block_start} block end (not included) {edit.block_end} "
                    f"actual_op_block_num {result.actual_op_block_num}"
                )
            else:
                logger.info(
                    f"Context management result: session {session_id} type {edit.type} "
                    f"status {result.op_status} "
                    f"block start {edit.block_start} block end (not included) {edit.block_end} "
                    f"reason {result.fail_reason}"
                )

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------
    def _execute_single_edit(
        self,
        edit: ContextManagementEditsParams,
        session_id: str,
    ) -> EditResponse:
        """执行单个 edit，通过注册的回调执行实际操作。"""
        # block_start / block_end 由 pymotor 从 message index 转换而来
        # 返回格式：EditResponse,记录op操作，session id，状态及详细信息
        if edit.target != "session" and (edit.block_start is None or edit.block_end is None):
            logger.info("Edit type=%s has no block_start/block_end, skipping. edit=%s", edit.type, edit)
            return EditResponse(
                session_id=session_id,
                type=edit.type,
                op_status=False,
                fail_reason=f"invalid block start {edit.block_start} or block end {edit.block_end}",
            )

        logger.info(
            f"Executing edit type={edit.type} for session {session_id}"
            f"(block_start={edit.block_start}, block_end={edit.block_end})."
        )

        callback_name = f"execute_{edit.type}"
        fn = self._registry.get(callback_name)
        if fn is None:
            logger.warning("No callback registered for edit type: %s, skipping.", edit.type)
            return EditResponse(
                session_id=session_id, type=edit.type, op_status=False, fail_reason=f"invalid edit type {edit.type}"
            )

        actual_process_blocks = 0
        op_result = True
        fail_reason = ""
        is_session_op = edit.target == "session"

        # TODO: 获取当前session 基础hash总长度
        total_block_length = len(self._registry.get("get_block_hashes_by_session")(session_id, 0, -1))
        logger.info(f"Process session {session_id} with block hash lengh {total_block_length}")
        op_result, fail_reason = self.process_edit_index(edit, total_block_length, session_id)

        if op_result:
            get_block_info_fn = None
            if edit.type == "prefetch":
                callback_name = "get_block_hashes_by_session"
                get_block_info_fn = self._registry.get(callback_name)
            else:
                callback_name = "get_global_block_id_by_session"
                get_block_info_fn = self._registry.get(callback_name)
            if get_block_info_fn is not None:
                process_session_block_info = get_block_info_fn(session_id, edit.block_start, edit.block_end)
                process_num = 0
                for info in process_session_block_info:
                    process_num += len(info)
                logger.info(f"session {session_id} session hash or block id count {process_num}")
                if edit.type == "prefetch":
                    actual_process_blocks = fn(session_id, process_session_block_info, is_session_op)
                else:
                    actual_process_blocks = fn(
                        session_id, edit.block_start, edit.block_end, process_session_block_info, is_session_op
                    )

        return EditResponse(
            session_id=session_id,
            type=edit.type,
            op_status=op_result,
            expected_op_block_num=edit.block_end - edit.block_start,
            actual_op_block_num=actual_process_blocks,
            fail_reason=fail_reason,
        )

    def process_edit_index(
        self, edit: ContextManagementEditsParams, candidate_list_length: int, session_id: str
    ) -> tuple[bool, str]:
        if edit.block_start is None:
            edit.block_start = 0

        if edit.block_end is None:
            edit.block_end = candidate_list_length
        else:
            # 左闭右闭
            edit.block_end += 1

        if candidate_list_length == 0:
            edit.block_start = edit.block_end = 0
            fail_reason = f"session_id {session_id} has {candidate_list_length} hash/block in SAM, fail to perform edit {edit.type}"
            logger.warning(
                f"Could not find session {session_id} with block ref record in SAM, failed to perform context management edit"
            )
            return (False, fail_reason)

        if edit.block_end < edit.block_start:
            fail_reason = f"block start {edit.block_start} is larger or equal to block end {edit.block_end}"
            edit.block_start = edit.block_end = 0
            return (False, fail_reason)

        if edit.block_start > candidate_list_length:
            logger.warning(
                f"edit index out of range: block end {edit.block_end} or block start {edit.block_start} "
                f"is out of index, the total kv length is {candidate_list_length}, fail to perform edit {edit.type}"
            )
            fail_reason = (
                f"block start {edit.block_start} or block end {edit.block_end} out of range {candidate_list_length}"
            )
            edit.block_start = edit.block_end = 0
            return (False, fail_reason)

        return (True, "")


def compute_ephemeral_range(
    cache_control: CacheControlParams,
) -> EphemeralRange | None:
    if cache_control is None:
        return None
    block_offset = cache_control.block_offset or 0  # 默认所有 block 都受保护
    return EphemeralRange(
        block_offset=block_offset,
        ttl=cache_control.ttl,
    )
