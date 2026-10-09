# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, G004
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from collections import OrderedDict

from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
)
from vllm.v1.request import Request

from vllm_ascend.core.agent_hint.params import (
    CacheControlParams,
    ContextManagementEditsParams,
    ContextManagementParams,
    get_agent_hint_params,
)
from vllm_ascend.core.agent_hint.session_event_listener import SessionEventListener

# Env var switch: when enabled, offload registers a 1h TTL protection on
# the affected blocks. Disabled by default (no TTL registered on offload).
_ENABLE_OFFLOAD_TTL_PROTECT = os.getenv("VLLM_ASCEND_OFFLOAD_TTL_PROTECT",
                                          "false").lower() in ("true", "1")


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
        # 只维护当前"请求"的hash表 — 与 SSM 共享同一份 dict 引用。
        # SAM 是这份 dict 的权威：register/evict 时由 SAM 写入/删除；
        # SSM 通过 _force_free_session_fn 在 LRU evict 时通知 SAM 删除 entry。
        self._session_block_hash: dict[str, list[BlockHash]] = {}

        # TTL 管理器（定时轮）
        self._ttl_manager = TTLManager(on_expired=self._on_ttl_expired, notify_func=self._notify_event)

        # 事件监听器列表
        self._event_listeners: list[SessionEventListener] = []

        self._session_state_manager = SessionStateManager(
            max_sessions=10,
            session_hash_state=self._session_block_hash,  # TODO: rename to self._session_hash_state
            hash_state={},                             # TODO: self._hash_state (SAM 侧待新增)
            apply_evict_remote_cache_fn=lambda hashes: self._notify_event("context_management_evict", block_hashes=hashes),
            apply_evict_kvblock_fn=self.apply_evict_kvblock,
        )

        # Session 控制器
        self._session_controller = SessionController(
            execute_offload=self._execute_offload,
            execute_prefetch=self._execute_prefetch,
            execute_evict=self._execute_evict,
            get_global_block_id_by_session=self._get_session_global_block_ids,
            get_block_hashes_by_session=self._get_session_block_hash,
            ssm_process_offload=self._session_state_manager.classify_offload_work,
            ssm_process_evict=self._session_state_manager.classify_evict_work,
          )

    def on_blocks_allocated_for_request(
        self, request: Request, blocks: KVCacheBlocks, cached_blocks_len_before: tuple[int, ...] | None = None
    ) -> None:
        # 解析引擎层透传的 opaque dict 为 ascend 强类型视图
        # 结果已缓存到 request._agent_hint_params，避免每个 block 都重新解析。
        hint = get_agent_hint_params(request)
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
            f"request.num_output_tokens: {request.num_output_tokens}"
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
        hint = get_agent_hint_params(request)
        if hint is None or hint.session_id is None:
            return

        logger.info(
            f"on_block_cache_hit_for_request, request_id = {request.request_id}, "
            f"session_id = {hint.session_id if hint else None}, "
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
            logger.debug(f"on_blocks_allocated, block_ids={block_ids}")
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
                    logger.debug("Newly cached block %s has no block hash", block_id)
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
                    newly_protected_hashes.append((block_id, block.block_hash))
                    newly_protected_ttl = ttl_expire_at

                # get_new_blocks，已经把物理 block 的 session 状态清零。
                # 使用真实 metadata 做校准，可以兼容异常重复回调：
                # 两个计数同时校准为 1（新引用建立）。
                current_ref_count = block.num_session_refs
                current_session_aware_ref = block.session_aware_ref
                self._update_block_meta(
                    block_id,
                    delta_ref=1 - current_ref_count,
                    delta_session_aware_ref=1 - current_session_aware_ref,
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
                        newly_protected_hashes.append((block_id, block.block_hash))
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
                        newly_protected_hashes.append((block_id, block.block_hash))
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
                        delta_session_aware_ref=+1,
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
        logger.debug(f"working on _on_ttl_expired in SAM for block {block_id} and session id {session_id}")

        # 统一接口：TTL 到期只清除 block 的 TTL，不动两个引用计数
        # （session record 仍在，num_session_refs 保持为 _block_sessions
        # 的投影）。这会导致 block 的 is_ephemeral() 返回 False，
        # FreeKVCacheBlockQueue.on_block_meta_changed 自动重评估分区，
        # 剩余引用使 block 落入 B 区。
        self._update_block_meta(
            block_id,
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

                    # 显式驱逐：两个计数同时 -1（_free_session 的两个调用方
                    # —— _free_session_tree（context_management evict）与
                    # _force_free_session（SSM LRU 驱逐）—— 都是驱逐语义）。
                    self._update_block_meta(
                        block_id,
                        delta_ref=-1,
                        delta_session_aware_ref=-1,
                        ttl_expire_at=latest_ttl_expire_at,
                    )

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

    def _force_free_session(self, session_id: str) -> list[BlockHash]:
        """SSM-facing wrapper: free a single session and its session-level TTL."""
        block_hashes = self._free_session(session_id)
        self._ttl_manager.remove(block_id=-1, session_id=session_id)
        return block_hashes

    def _free_session_tree(self, session_id: str) -> list[BlockHash]:
        """Recursively free a session and all its children."""
        logger.info("SAM: _free_session_tree [%s]", session_id)
        session_info = self._sessions.get(session_id)
        if session_info is None:
            return []

        all_hashes: list[BlockHash] = []
        for child_sid in list(self._sessions[session_id].children):
            all_hashes.extend(self._free_session_tree(child_sid))

        parent_hashes = self._free_session(session_id)
        all_hashes.extend(parent_hashes)

        logger.debug(
            f"Free session tree for session {session_id}: "
            f"total_hashes={len(all_hashes)}, "
            f"session_info: {self._sessions}, "
        )

        return parent_hashes

    def _execute_offload(
        self,
        session_id: str,
        block_start: int,
        block_end: int,
        block_ids_all_group: tuple[list[int], ...],
    ) -> int:
        """卸载指定范围的 block — 减少 session 引用 + 清除当前session的TTL（通知TTLManager）,仅操作 SAM 本地 HBM 数据。"""
        affected_block_hashes: list[BlockHash] = []
        update_count = 0
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

                self._remove_session_block_ref(session_id, block_id, group_id)

                remaining_records = self._block_sessions[group_id].get(block_id, {}).values()
                latest_ttl_expire_at = max(
                    (remaining_record.ttl_expire_at for remaining_record in remaining_records),
                    default=0.0,
                )
                # offload 只减少当前引用，不动 session_aware_ref：
                # block 仍留在 B 区，直到显式 evict 消耗该计数。
                self._update_block_meta(
                    block_id,
                    delta_ref=-1,
                    ttl_expire_at=latest_ttl_expire_at,
                )
                update_count += 1

        if update_count:
            block_start_e, block_end_e = self._expand_session_block_range(session_id, block_start, block_end)
            affected_block_hashes.extend(self._session_block_hash.get(session_id, [])[block_start_e:block_end_e])
        res = len(affected_block_hashes)

        if _ENABLE_OFFLOAD_TTL_PROTECT:
            self._ttl_manager.register(
                block_infos=[],
                session_id=session_id,
                expire_at=time.monotonic() + 3600,
                protected_block_hashes=affected_block_hashes,
            )

        return res

    def _execute_prefetch(self, session_id: str, block_hashes: list[BlockHash]) -> int:
        """通知 SPM 创建远端预取任务。"""
        self._notify_event(
            "context_management_prefetch",
            session_id=session_id,
            block_hashes=block_hashes,
        )
        return len(block_hashes)

    def _execute_evict(
        self,
        session_id: str,
        block_start: int,
        block_end: int,
        block_ids_all_group: tuple[list[int], ...],
        is_session: bool,
    ) -> list[BlockHash]:
        """Evict blocks — drop session ref + clear TTL."""
        if is_session:
            all_hashes = self._free_session_tree(session_id)
            self._ttl_manager.remove(block_id=-1, session_id=session_id)
            return all_hashes

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

                self._remove_session_block_ref(session_id, block_id, group_id)

                remaining_records = self._block_sessions[group_id].get(block_id, {}).values()
                latest_ttl_expire_at = max(
                    (remaining_record.ttl_expire_at for remaining_record in remaining_records),
                    default=0.0,
                )
                # 显式 block-range evict：两个计数同时 -1。record 存在才走到
                # 这里，重复 evict 时 record 已被移除、直接 continue，保持幂等。
                self._update_block_meta(
                    block_id,
                    delta_ref=-1,
                    delta_session_aware_ref=-1,
                    ttl_expire_at=latest_ttl_expire_at,
                )

        block_start_e, block_end_e = self._expand_session_block_range(session_id, block_start, block_end)
        old_hashes = self._session_block_hash.get(session_id, [])
        affected_block_hashes.extend(old_hashes[block_start_e:block_end_e])
        # Update shared_dict: drop evicted hashes so the dict stays in sync
        # with the actual state. SSM reads the updated list through the
        # shared reference.
        if old_hashes:
            self._session_block_hash[session_id] = (
                old_hashes[:block_start_e] + old_hashes[block_end_e:]
            )

        self._ttl_manager.remove(block_id=-1, session_id=session_id)
        return affected_block_hashes

    def _get_session_global_block_ids(
        self,
        session_id: str,
        block_start: int,
        block_end: int,
    ) -> tuple[tuple[list[int], ...], list[BlockHash]]:
        """将基础 hash block 范围映射为各 KV cache group 的物理 block id。

        当某个 KV cache group 的 block size 大于 hash_block_size 时，只要
        物理 block 与指定逻辑范围有重叠，就返回该物理 block。

        返回 (per-group block id tuple, hash list)。hash list 按
        hash_block_size 粒度切片，范围是基于 self.block_size 配置算出的
        所有 group 实际覆盖范围的并集 [min(start), max(end))（不依赖
        session_blocks 状态，确保多次 offload 后 session_blocks 已被清空
        时仍能正确切出 hash list —— offload 不清 shared_dict）。每个 group
        的切片都是 base_block_hashes 的连续子区间，并集等价于
        [min(start), max(end))，直接切 base 即保序。block_ids 部分依赖
        session_blocks，session_blocks 已被清空时 result[group_id] 为空。
        """
        result: tuple[list[int], ...] = tuple([] for _ in range(self.num_kv_cache_groups))

        base_block_hashes = self._session_block_hash.get(session_id)
        if not base_block_hashes:
            logger.warning(f"There is no hash for the session_id:{session_id} in the _session_block_hash.")
            return result, []

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
            return result, []

        block_end = min(block_end, base_block_count)

        if block_start >= block_end:
            return result, []

        union_start = block_end
        union_end = block_start

        for group_block_size in self.block_size:
            assert group_block_size % self.hash_block_size == 0, (
                f"block_size {group_block_size} is not a multiple of {self.hash_block_size}"
            )
            scale_factor = group_block_size // self.hash_block_size
            group_block_start = block_start // scale_factor
            group_block_end = (block_end + scale_factor - 1) // scale_factor
            union_start = min(union_start, group_block_start * scale_factor)
            union_end = max(union_end, group_block_end * scale_factor)
        if union_start < union_end:
            longest_hash_list = list(base_block_hashes[union_start:union_end])
        else:
            longest_hash_list = []

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

        return result, longest_hash_list

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

    def process_prefetch_req(self, num_unfinished_requests: int) -> None:
        """统一 prefetch 入口：调 SPM 处理预取，然后更新 SSM hash 位置为 HBM。"""
        for listener in tuple(self._event_listeners):
            handler = getattr(listener, "process_prefetch_req", None)
            if handler is not None:
                results = handler(num_unfinished_requests)
                for session_id, block_hashes in results:
                    self._session_state_manager.process_prefetch_hashes(session_id, block_hashes)
                break

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

    def on_request_completed(self, request_id: str, session_id: str | None) -> None:
        """请求完成回调 — 处理挂起 edits 并通知 SSM 会话不再活跃"""
        self._session_controller.on_request_completed(request_id)
        if session_id:
            self._session_state_manager.mark_session_inactive(session_id)

    def tick(self, now: float | None = None) -> None:
        """推进 TTL timer wheel，并在 LRU 满时尝试淘汰 session。由 scheduler 统一调用。"""
        self._ttl_manager.tick(now)
        self._session_state_manager._try_evict_if_needed()

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
        """register session block hash to session aware manager and session state manager.
        """
        self._session_state_manager.set_session_block_hashes(session_id, block_hashes)
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
        delta_session_aware_ref: int = 0,
        ttl_expire_at: float | None = None,
    ) -> None:
        """Update Agent Hint metadata and re-evaluate the free queue zone.

        ``delta_ref`` adjusts the current session reference count;
        ``delta_session_aware_ref`` adjusts the session-aware count that is
        only consumed by explicit evicts (it replaces the old
        ``is_offload_block`` flag).  Both are clamped at 0 defensively —
        an underflow means a double evict/remove, which is logged instead
        of silently swallowed.
        """
        block_pool = self.kv_cache_manager.block_pool
        block = block_pool.blocks[block_id]
        session_ref_cnt = None
        if delta_ref != 0:
            new_ref = block.num_session_refs + delta_ref
            if new_ref < 0:
                logger.warning(
                    "_update_block_meta: session ref underflow on block %s "
                    "(%d + %d), clamping to 0 — possible double evict/remove",
                    block_id, block.num_session_refs, delta_ref,
                )
            session_ref_cnt = max(0, new_ref)
        session_aware_ref = None
        if delta_session_aware_ref != 0:
            new_aware_ref = block.session_aware_ref + delta_session_aware_ref
            if new_aware_ref < 0:
                logger.warning(
                    "_update_block_meta: session_aware_ref underflow on block %s "
                    "(%d + %d), clamping to 0 — possible double evict",
                    block_id, block.session_aware_ref, delta_session_aware_ref,
                )
            session_aware_ref = max(0, new_aware_ref)
        block.set_session_state(
            session_ref_cnt=session_ref_cnt,
            session_aware_ref=session_aware_ref,
            ttl_expire_at=ttl_expire_at,
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

    def apply_offload_kvblock(
        self,
        hbm_work: list[BlockHashState],
    ) -> None:
        """SAM 侧 HBM apply for offload (TODO 待实现)."""
        return

    def apply_evict_kvblock(
        self,
        session_id: str,
        work: list[tuple[BlockHashState, list[int]]],
        is_session_op: bool,
    ) -> list[BlockHash]:
        """SAM 侧 HBM + TTL apply for evict (TODO 待实现).

        Signature matches ``apply_evict_kvblock_fn`` callback expected
        by SSM. Prefixed with ``_`` because it's bound as a callback
        (not meant to be called directly by callers other than SSM).
        """
        return []


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
        if protected_block_hashes is not None and len(protected_block_hashes) > 0:
            block_infos.append((0, protected_block_hashes))
            protected_hash_len = len(protected_block_hashes)
        else:
            return

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
    """Executor for ``context_management.edits``.

    Processes the ``context_management.edits`` carried by a request,
    executing them immediately or deferring them until the request
    finishes based on the ``manage_request`` flag. It performs the actual
    block operations through registered callbacks and is agnostic to the
    concrete offload/prefetch/evict implementations.

    The callbacks come in two categories:

    Execute callbacks (registered in ``_execute_sam_callbacks`` /
    ``_execute_ssm_callbacks``, keyed by edit type and looked up per edit)
    ----------------------------------------------------------------------
    SAM-side callbacks (operate on local HBM blocks):
      ``execute_offload``   (session_id, block_start, block_end, block_ids_all_group) -> int
      ``execute_prefetch``  (session_id, block_hashes) -> int
      ``execute_evict``     (session_id, [block_start, block_end, block_ids_all_group,] is_session) -> list[BlockHash]
                            Returns flat list of evicted hashes. Range evict computes
                            affected hashes via ``_expand_session_block_range`` internally.
                            Session evict returns flat list from ``_free_session_tree``
                            (children + parent). SSM processes all hashes under the single
                            top-level session_id.
    SSM-side callbacks (update remote KV pool state):
      ``ssm_process_offload`` (session_id, block_hashes) -> None
      ``ssm_process_evict``   (session_id, block_hashes, is_session) -> None

    Query callbacks (stored as attributes, invoked directly)
    --------------------------------------------------------
    ``get_block_hashes_by_session``  (session_id, block_start, block_end) -> list[BlockHash]
    ``get_global_block_id_by_session`` (session_id, block_start, block_end) -> tuple[tuple[list[int], ...], list[BlockHash]]
        Returns (per-group block ids, longest hash list at hash_block_size
        granularity). The hash list is forwarded to SAM callbacks and SSM
        so all three see the same hash — avoids the previous mismatch where
        SAM sliced shared_dict itself and could include hashes for unhashed
        blocks that ``block_ids_all_group`` skipped.

    Note:
      - Blocks/indices are NOT resolved here; each SAM callback resolves the
        logical range into physical blocks through the query callbacks.
      - ``prefetch`` is resolved via ``get_block_hashes_by_session`` and
        forwards the block hashes to the SPM; other edit types resolve via
        ``get_global_block_id_by_session`` and hand over
        ``(block_start, block_end, physical block ids, hash list, is_session)``.
    """

    def __init__(
        self,
        *,
        execute_offload: Callable[..., int],
        execute_prefetch: Callable[..., int],
        execute_evict: Callable[..., list[BlockHash]],
        get_global_block_id_by_session: Callable[[str, int, int], tuple[tuple[list[int], ...], list[BlockHash]]],
        get_block_hashes_by_session: Callable[[str, int, int], list[BlockHash]],
        ssm_process_offload=Callable[..., None],
        ssm_process_evict=Callable[..., None],

    ):
        self._execute_sam_callbacks: dict[str, Callable[..., Any]] = {
            "offload": execute_offload,
            "prefetch": execute_prefetch,
            "evict": execute_evict,
        }
        self._execute_ssm_callbacks: dict[str, Callable[..., None]] = {
            "offload": ssm_process_offload,
            "evict": ssm_process_evict,
        }
        self._get_global_block_id_by_session = get_global_block_id_by_session
        self._get_block_hashes_by_session = get_block_hashes_by_session

        # request_id → [(edit, session_id), ...]
        self._pending_edits: dict[str, list[tuple[ContextManagementEditsParams, str]]] = {}

        logger.info("SessionController initialized.")

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

        sam_fn = self._execute_sam_callbacks.get(edit.type)
        if sam_fn is None:
            logger.warning("No callback registered for edit type: %s, skipping.", edit.type)
            return EditResponse(
                session_id=session_id, type=edit.type, op_status=False, fail_reason=f"invalid edit type {edit.type}"
            )

        actual_process_blocks = 0
        op_result = True
        fail_reason = ""
        is_session_op = edit.target == "session"

        # TODO: 获取当前session 基础hash总长度
        total_block_length = len(self._get_block_hashes_by_session(session_id, 0, -1))
        logger.info(f"Process session {session_id} with block hash lengh {total_block_length}")
        op_result, fail_reason = self.process_edit_index(edit, total_block_length, session_id)

        if op_result:
            hashes_for_ssm: list[BlockHash] | None = None

            if edit.type == "prefetch":
                block_hashes_for_sam = self._get_block_hashes_by_session(
                    session_id, edit.block_start, edit.block_end)
                actual_process_blocks = sam_fn(session_id, block_hashes_for_sam)
            else:
                # evict / offload: resolve global block ids + union hash list,
                # 透传给 SAM 回调。evict 把 is_session_op 一并透传，SAM 内
                # 部根据该 flag 走 _free_session_tree 或 range 分支；offload
                # 不需要该 flag。
                process_session_block_info, block_hashes = self._get_global_block_id_by_session(
                    session_id, edit.block_start, edit.block_end)
                process_num = sum(len(info) for info in process_session_block_info)
                logger.info(f"session {session_id} block id count {process_num}")

                if edit.type == "evict":
                    evicted_hashes = sam_fn(
                        session_id, edit.block_start, edit.block_end,
                        process_session_block_info, is_session_op)
                    actual_process_blocks = len(evicted_hashes)
                    hashes_for_ssm = evicted_hashes
                else:  # offload
                    actual_process_blocks = sam_fn(
                        session_id, edit.block_start, edit.block_end,
                        process_session_block_info)
                    hashes_for_ssm = block_hashes

                ssm_fn = self._execute_ssm_callbacks.get(edit.type)
                if ssm_fn is not None:
                    #TODO: 传入根据边界切割后的state list
                    states_for_ssm = [
                        st for h in hashes_for_ssm
                        if (st := self._session_state_manager._hash_state.get(h)) is not None
                    ]
                    ssm_fn(session_id, states_for_ssm)

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

@dataclass
class BlockHashState:
    """Complete state of a single block hash — reference tracking + offload phase counter.

    SSM is agnostic about the concrete cache location of a hash; it only
    maintains an offload phase counter:
    - 0: not offloaded (HBM)
    - 1: offloaded one level
    - 2: sunk to bottom, reached the cap, cannot offload further (entering
      this phase triggers SPM notification to release remote data)

    ``block_ids_per_group`` lists the physical block ids that currently
    hold this hash across all KV cache groups. SSM itself does not consume
    this field for ref/offload bookkeeping; it is metadata for upstream
    layers (SAM/SPM) to locate the concrete blocks when an offload/evict
    is materialized. Maintained by SAM on alloc/cache-hit/free.
    """
    block_hash: BlockHash
    ref_count: int = 0
    ref_sessions: set[str] = field(default_factory=set)
    offload_count: int = 0
    max_offload_count: int = 2
    # Physical block ids holding this hash across all groups. Duplicates
    # are ignored; removal is by value.
    block_ids_per_group: list[int] = field(default_factory=list)

    def add_ref(self, session_id: str) -> bool:
        """Add a session reference. Returns whether offload_count was reset."""
        self.ref_count += 1
        self.ref_sessions.add(session_id)
        if self.offload_count != 0:
            self.offload_count = 0
            return True  # was reset from offloaded state back to HBM
        return False  # already in HBM (offload_count == 0)

    def remove_ref(self, session_id: str) -> bool:
        """Remove a session reference. Returns True when fully released (ref_count hit 0)."""
        self.ref_count -= 1
        self.ref_sessions.discard(session_id)
        return self.ref_count == 0

    def offload_once(self) -> bool:
        """Advance one offload phase. Returns whether SPM must be notified
        (transition 1 -> 2).

        0 -> 1: internal state advance only, no SPM notification
                (data may still be in local cache).
        1 -> 2: sunk to bottom; SPM must be notified to actually release
                the remote data.
        Already at 2 (>= max): cannot offload further, returns False.
        """
        if self.offload_count >= self.max_offload_count:
            return False
        self.offload_count += 1
        return self.offload_count == self.max_offload_count

    def is_shared_with_active_session(self, requesting_session: str, is_active_fn) -> bool:
        """Check whether the hash is shared by another active session
        (multi-session protection check).
        """
        if self.ref_count <= 1:
            return False
        other_sessions = self.ref_sessions - {requesting_session}
        return any(is_active_fn(sid) for sid in other_sessions)

    def add_block_id(self, block_id: int) -> None:
        """Register a physical block holding this hash.

        Idempotent: duplicate calls with the same block_id do not append
        a second entry. SAM calls this on alloc / cache-hit once the
        block↔hash mapping is established.
        """
        if block_id not in self.block_ids_per_group:
            self.block_ids_per_group.append(block_id)

    def remove_block_id(self, block_id: int) -> bool:
        """Unregister a physical block.

        Returns True if the block was present and removed; False if it
        was not found (no-op, logged at debug level — happens on
        duplicate free or when the hash was never materialized in that
        block).
        """
        try:
            self.block_ids_per_group.remove(block_id)
            return True
        except ValueError:
            logger.warning(
                "BlockHashState.remove_block_id: block %s not found for hash %s",
                block_id, self.block_hash,
            )
            return False


class SessionLRU:
    """Session LRU container — pure data structure, unaware of hash-level state.

    Maintains two indexes:
      - ``_lru``: OrderedDict[sid -> last_touch_ts], iteration order is the LRU
        order (older entries first)
      - ``_session_hash_count``: sid -> hash count, used as the evict-candidate
        ordering key (fewer hashes evicted first)

    Candidate selection rule: exclude active sessions -> sort by hash_count
    ascending -> ties keep LRU order.
    """

    def __init__(self, max_sessions: int = 1000):
        self._lru: OrderedDict[str, float] = OrderedDict()
        self._session_hash_count: dict[str, int] = {}
        self.max_sessions = max_sessions

    def touch(self, session_id: str) -> None:
        """Move session to the end (most recently used)."""
        if session_id in self._lru:
            self._lru.move_to_end(session_id)
        else:
            self._lru[session_id] = time.monotonic()

    def remove(self, session_id: str) -> None:
        """Drop session from both LRU and hash-count index."""
        self._lru.pop(session_id, None)
        self._session_hash_count.pop(session_id, None)

    def set_hash_count(self, session_id: str, count: int) -> None:
        """Update hash count used as evict ordering key."""
        self._session_hash_count[session_id] = count

    def is_managed(self, session_id: str) -> bool:
        return session_id in self._lru

    def is_full(self) -> bool:
        return len(self._lru) >= self.max_sessions

    def get_evict_candidates(
        self,
        count: int = 1,
        active_session_ids: set[str] | None = None,
    ) -> list[str]:
        """Return up to ``count`` inactive sessions, least-hash-count first.

        Ties on hash_count preserve LRU order (stable sort keeps older
        entries first).
        """
        active = active_session_ids or set()
        candidates = [sid for sid in self._lru if sid not in active]
        candidates.sort(key=lambda sid: self._session_hash_count.get(sid, 0))
        return candidates[:count]

    def __len__(self) -> int:
        return len(self._lru)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._lru


class SessionStateManager:
    """Session state manager — hash 维度状态权威 + LRU 驱逐自编排。

    SSM 共享 SAM 的两张表（构造时传入同一份 dict 引用）：
      - ``_session_hash_state: dict[sid, list[BlockHashState]]``：session →
        BlockHashState 列表（保序）。SAM 写入初始 list，SSM 负责
        trim / pop（evict / LRU 驱逐时）。
      - ``_hash_state: dict[BlockHash, BlockHashState]``：hash → state
        全局反查。SAM 创建 entry（register 时），SSM 负责增删 entry
        + 改字段（ref_count / ref_sessions / offload_count）。

    SSM 是 state 的权威维护者：分类时读 state 字段做分类，同时改
    state（ref_count / ref_sessions / offload_count / 增删 _hash_state /
    trim / pop _session_hash_state）。SSM 不直接改 HBM block
    （``block.sessions`` / ``set_session_state``），HBM apply 通过
    callback 调 SAM 完成。

    SSM 不感知父子 session —— 只处理传入的单个 ``session_id``。LRU
    驱逐也只处理单个 top-level session，children 由上层维护。
    """

    def __init__(
        self,
        max_sessions: int = 1000,
        session_hash_state: dict[str, list[BlockHashState]] | None = None,
        hash_state: dict[BlockHash, BlockHashState] | None = None,
        apply_evict_remote_cache_fn: Callable[[list[BlockHash]], int | None] | None = None,
        apply_evict_kvblock_fn: Callable[[str, list[tuple[BlockHashState, list[int]]], bool], list[BlockHash]] | None = None,
    ):
        self._session_hash_state: dict[str, list[BlockHashState]] = (
            session_hash_state if session_hash_state is not None else {}
        )
        self._hash_state: dict[BlockHash, BlockHashState] = (
            hash_state if hash_state is not None else {}
        )
        self._lru = SessionLRU(max_sessions=max_sessions)
        self._active_session_ids: set[str] = set()

        self._apply_evict_remote_cache_fn = apply_evict_remote_cache_fn
        self._apply_evict_kvblock_fn = apply_evict_kvblock_fn

    # ════════════════════════════════════════════════════════════════
    # Active session tracking
    # ════════════════════════════════════════════════════════════════

    def mark_session_active(self, session_id: str) -> None:
        if session_id not in self._active_session_ids:
            logger.info("SSM: mark_session_active [%s]", session_id)
            self._active_session_ids.add(session_id)

    def mark_session_inactive(self, session_id: str) -> None:
        if session_id in self._active_session_ids:
            logger.info("SSM: mark_session_inactive [%s]", session_id)
            self._active_session_ids.discard(session_id)

    def _is_session_active(self, session_id: str) -> bool:
        return session_id in self._active_session_ids

    def set_session_block_hashes(
        self, session_id: str, new_hashes: list[BlockHash],
    ) -> None:
        """Diff-register hashes: update BlockHashState ref_count/ref_sessions.

        Diff strategy (only handle added/removed, leave common alone):
          - removed (old - new): decrement ref_count; if 0, del _hash_state.
          - added   (new - old): create state or add_ref (resets offload_count).
          - common  (old & new): untouched — avoids resetting offload_count
            of a long-held hash or triggering state delete-then-recreate.
        """
        # TODO:确认sam调用的时间点及_session_hash_state的更新，目前接口暂未更新_session_hash_state
        old_states = self._session_hash_state.get(session_id, [])
        old_hashes = {s.block_hash for s in old_states}
        new_set = set(new_hashes)

        removed = [s for s in old_states if s.block_hash not in new_set]
        added = [h for h in new_hashes if h not in old_hashes]
        if removed or added:
            logger.info("SSM: set_session_block_hashes [%s] %d -> %d hashes (added=%d, removed=%d)",
                         session_id, len(old_states), len(new_hashes), len(added), len(removed))

        for state in removed:
            if state.remove_ref(session_id):
                del self._hash_state[state.block_hash]

        new_count = 0
        reset_count = 0
        for h in added:
            state = self._hash_state.get(h)
            if state is not None:
                if state.add_ref(session_id):
                    reset_count += 1
            else:
                self._hash_state[h] = BlockHashState(
                    block_hash=h, ref_count=1, ref_sessions={session_id},
                    offload_count=0,
                )
                new_count += 1

        logger.info("SSM: set_session_block_hashes [%s] %d new global entries, %d reset to HBM, total managed hashes=%d",
                      session_id, new_count, reset_count, len(self._hash_state))
        self._lru.set_hash_count(session_id, len(new_hashes))
        self.mark_session_active(session_id)
        self._lru.touch(session_id)

    # ════════════════════════════════════════════════════════════════
    # Classify offload work — Controller 调用
    # ════════════════════════════════════════════════════════════════

    def classify_offload_work(
        self, session_id: str, states: list[BlockHashState],
    ) -> dict[str, list]:
        """Classify states for offload + advance offload_count at classify time.

        State attributes are updated here (SSM owns them): ``offload_count``
        is advanced via ``offload_once()``, and ``block_ids_per_group`` is
        cleared after the offload action (HBM no longer holds these blocks).

        Returns OffloadWork dict:
          - ``hbm_work``:   [(state, block_ids_per_group_snapshot) for state in first_offload_states]
                            offload_count 0→1, Controller → sam.apply_offload_kvblock
          - ``remote_work``: [state.block_hash for state in second_offload_states]
                            offload_count 1→2, Controller → spm.on_context_management_offload
          - ``skipped``:    [state for state in active_shared_or_max_reached]
                            no change
        """
        hbm_work: list[tuple[BlockHashState, list[int]]] = []
        remote_work: list[BlockHash] = []
        skipped: list[BlockHashState] = []
        for state in states:
            if state.is_shared_with_active_session(session_id, self._is_session_active):
                skipped.append(state)
                continue
            if state.offload_count >= state.max_offload_count:
                skipped.append(state)
                continue
            reached_bottom = state.offload_once()
            # offload 后 HBM 不再持有这些 block，清空 state.block_ids_per_group
            block_ids_snapshot = list(state.block_ids_per_group)
            state.block_ids_per_group = []
            if reached_bottom:
                remote_work.append(state.block_hash)
            else:
                hbm_work.append((state, block_ids_snapshot))

        logger.info("SSM: classify_offload_work [%s] hbm=%d remote=%d skipped=%d",
                     session_id, len(hbm_work), len(remote_work), len(skipped))
        self._lru.touch(session_id)
        return {"hbm_work": hbm_work, "remote_work": remote_work, "skipped": skipped}

    # ════════════════════════════════════════════════════════════════
    # Classify evict work — Controller 调用
    # ════════════════════════════════════════════════════════════════

    def classify_evict_work(
        self,
        session_id: str,
        states: list[BlockHashState],
    ) -> dict[str, list[tuple[BlockHashState, list[int]]]]:
        """Classify states for evict + mutate state at classify time.

        Returns EvictWork dict:
          - ``hbm_only_work``:      [(state, block_ids_per_group) for state in still_shared_states]
                                    ref_sessions 还有其他 session（非 active）, 仅减 ref (state 保留)
          - ``hbm_and_remote_work``: [(state, block_ids_per_group) for state in exclusive_states]
                                    ref_sessions 只剩当前 sid, SSM 已删 state entry
          - ``skipped``:            [state for state in active_share_protected]
                                    no change
          Each work entry is ``(state, block_ids_snapshot)`` — snapshot
          taken before the block list is cleared, so callers can still
          locate the concrete HBM blocks to release.
        """
        hbm_only_work: list[tuple[BlockHashState, list[int]]] = []
        hbm_and_remote_work: list[tuple[BlockHashState, list[int]]] = []
        skipped: list[BlockHashState] = []
        for state in states:
            if state.is_shared_with_active_session(session_id, self._is_session_active):
                skipped.append(state)
                continue
            state.ref_sessions.discard(session_id)
            state.ref_count = max(0, state.ref_count - 1)
            block_ids_snapshot = list(state.block_ids_per_group)
            # 无 active 共享即可清空 block 归属（与 offload 一致），
            # 不必等 ref_count == 0。
            state.block_ids_per_group = []
            if state.ref_count == 0:
                del self._hash_state[state.block_hash]
                hbm_and_remote_work.append((state, block_ids_snapshot))
            else:
                hbm_only_work.append((state, block_ids_snapshot))

        # SSM bookkeeping on _session_hash_state + LRU.
        # is_session_op: states 覆盖了该 session 全部 state → 整 session evict。
        session_states = self._session_hash_state.get(session_id, [])
        is_session_op = len(states) >= len(session_states)
        if is_session_op:
            self._session_hash_state.pop(session_id, None)
            self._lru.remove(session_id)
        else:
            # range evict: 从 _session_hash_state[sid] 里剔除本轮 evict 的 state。
            # 用 block_hash 集合做差集，避免残留已被 del _hash_state 的僵尸引用。
            evicted_hashes = {s.block_hash for s in states}
            old = self._session_hash_state.get(session_id)
            if old is not None:
                self._session_hash_state[session_id] = [
                    s for s in old if s.block_hash not in evicted_hashes
                ]
            remaining = self._session_hash_state.get(session_id)
            if remaining:
                self._lru.touch(session_id)
                self._lru.set_hash_count(session_id, len(remaining))

        logger.info("SSM: classify_evict_work [%s] hbm_only=%d hbm_and_remote=%d skipped=%d (is_session=%s)",
                     session_id, len(hbm_only_work), len(hbm_and_remote_work), len(skipped), is_session_op)
        return {"hbm_only_work": hbm_only_work, "hbm_and_remote_work": hbm_and_remote_work, "skipped": skipped}

    # ════════════════════════════════════════════════════════════════
    # Prefetch — SPM 完成后调 SSM 复位 offload_count
    # ════════════════════════════════════════════════════════════════

    def process_prefetch_hashes(
        self, session_id: str, block_hashes: list[BlockHash],
    ) -> None:
        """Reset offload_count of the input hashes back to 0 (HBM)."""
        if not block_hashes:
            return
        reset_count = 0
        for h in block_hashes:
            state = self._hash_state.get(h)
            if state is not None and state.offload_count != 0:
                state.offload_count = 0
                reset_count += 1
        logger.info("SSM: process_prefetch_hashes [%s] reset %d / %d hashes to HBM",
                     session_id, reset_count, len(block_hashes))
        if self._lru.is_managed(session_id):
            self._lru.touch(session_id)

    def _try_evict_if_needed(self) -> None:
        """Proactively evict inactive sessions until the LRU is no longer full.

        Drives one session per iteration; re-checks ``is_full`` after each
        eviction so a multi-session overflow is fully drained in one call.
        Stops early if no inactive candidate is available (avoids spinning
        while all remaining sessions are active).
        """
        remaining_budget = len(self._lru)
        while self._lru.is_full() and remaining_budget > 0:
            remaining_budget -= 1
            candidates = self._lru.get_evict_candidates(
                count=1, active_session_ids=self._active_session_ids)
            if not candidates:
                return

            sid = candidates[0]
            logger.info("SSM: _try_evict_if_needed LRU full (%d >= %d), evicting [%s]",
                         len(self._lru), self._lru.max_sessions, sid)

            # 1. slice states (whole session)
            states = self._session_hash_state.get(sid, [])
            if not states:
                self._lru.remove(sid)
                continue

            # 2. classify + mutate state (pop _session_hash_state + remove LRU inside)
            evict_work = self.classify_evict_work(sid, states)
            hbm_only_work = evict_work["hbm_only_work"]
            hbm_and_remote_work = evict_work["hbm_and_remote_work"]

            # 3. SAM applies HBM + TTL
            freed_hashes: list[BlockHash] = []
            if self._apply_evict_kvblock_fn is not None:
                all_work = hbm_only_work + hbm_and_remote_work
                freed_hashes = self._apply_evict_kvblock_fn(
                    sid, all_work, True) or []

            # 4. notify SPM (freed = exclusive hashes, safe to delete remote)
            if self._apply_evict_remote_cache_fn is not None and freed_hashes:
                self._apply_evict_remote_cache_fn(freed_hashes)

            logger.info(
                "SSM: _try_evict_if_needed [%s] done — states=%d hbm_only=%d "
                "hbm_and_remote=%d skipped=%d freed_hashes=%d (spm_notified=%s)",
                sid, len(states), len(hbm_only_work), len(hbm_and_remote_work),
                len(evict_work.get("skipped", [])), len(freed_hashes),
                bool(freed_hashes) and self._apply_evict_remote_cache_fn is not None,
            )

