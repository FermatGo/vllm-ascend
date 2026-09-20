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
        # 只维护当前"请求"的hash表
        self._session_block_hash: dict[str, list[BlockHash]] = {}

        # TTL 管理器（定时轮）
        self._ttl_manager = TTLManager(on_expired=self._on_ttl_expired, notify_func=self._notify_event)

        # 事件监听器列表
        self._event_listeners: list[SessionEventListener] = []

        # session state manager
        num_cache_levels = int(os.environ.get("NUM_CACHE_LEVELS", 2))

        self._session_state_manager = SessionStateManager(
            max_sessions=10,
            on_offload=lambda hashes: self._notify_event("context_management_offload", block_hashes=hashes),
            on_evict=lambda hashes: self._notify_event("context_management_evict", block_hashes=hashes) if num_cache_levels == 3 else self._notify_event("context_management_offload", block_hashes=hashes),
        )

        # Session 控制器
        self._session_controller = SessionController(
            execute_offload=self._execute_offload,
            execute_prefetch=self._execute_prefetch,
            execute_evict=self._execute_evict,
            get_global_block_id_by_session=self._get_session_global_block_ids,
            get_block_hashes_by_session=self._get_session_block_hash,
            ssm_process_offload=self._session_state_manager.process_offload_hashes,
            ssm_process_evict=self._session_state_manager.process_evict_hashes,
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

        # 统一接口：清除 TTL + 减少 session 引用
        # 这会导致 block 的 is_ephemeral() 返回 False
        # FreeKVCacheBlockQueue.on_block_meta_changed 自动重评估分区
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
                self._update_block_meta(
                    block_id,
                    delta_ref=-1,
                    ttl_expire_at=latest_ttl_expire_at,
                    is_offload_block=True,
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

    def _execute_prefetch(self, session_id: str, block_hashes: list[BlockHash], is_session: bool = False) -> int:
        """通知 SPM 创建远端预取任务。"""
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
        """register session block hash to session aware manager and session state manager"""
        self._session_block_hash[session_id] = block_hashes
        self._session_state_manager.set_session_block_hashes(session_id, block_hashes)

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
    # TODO: consider recomputing hashes when a prefetch request has content
    #       or when the session has already been cleaned up
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
      ``execute_offload``   (session_id, block_start, block_end, block_ids_all_group, is_session) -> int
      ``execute_prefetch``  (session_id, block_hashes, is_session) -> int
      ``execute_evict``     (session_id, block_start, block_end, block_ids_all_group, is_session) -> int
    SSM-side callbacks (update remote KV pool state):
      ``ssm_process_offload`` (session_id, block_start, block_end, is_session) -> None
      ``ssm_process_evict``   (session_id, block_start, block_end, is_session) -> None

    Query callbacks (stored as attributes, invoked directly)
    --------------------------------------------------------
    ``get_block_hashes_by_session``  (session_id, block_start, block_end) -> list[BlockHash]
    ``get_global_block_id_by_session`` (session_id, block_start, block_end) -> tuple[list[int], ...]

    Note:
      - Blocks/indices are NOT resolved here; each SAM callback resolves the
        logical range into physical blocks through the query callbacks.
      - ``prefetch`` is resolved via ``get_block_hashes_by_session`` and
        forwards the block hashes to the SPM; other edit types resolve via
        ``get_global_block_id_by_session`` and hand over
        ``(block_start, block_end, physical block ids, is_session)``.
    """

    def __init__(
        self,
        *,
        execute_offload: Callable[..., int],
        execute_prefetch: Callable[..., int],
        execute_evict: Callable[..., int],
        get_global_block_id_by_session: Callable[[str, int, int], tuple[list[int], ...]],
        get_block_hashes_by_session: Callable[[str, int, int], list[BlockHash]],
        ssm_process_offload=Callable[..., None],
        ssm_process_evict=Callable[..., None],

    ):
        self._execute_sam_callbacks: dict[str, Callable[..., int]] = {
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

        fn = self._execute_sam_callbacks.get(edit.type)
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
        total_block_length = len(self._get_block_hashes_by_session(session_id, 0, -1))
        logger.info(f"Process session {session_id} with block hash lengh {total_block_length}")
        op_result, fail_reason = self.process_edit_index(edit, total_block_length, session_id)

        if op_result:
            # ── SSM 侧处理（远端状态更新） ──
            ssm_fn = self._execute_ssm_callbacks.get(edit.type)
            if ssm_fn is not None:
                ssm_fn(session_id, edit.block_start, edit.block_end,
                    is_session=is_session_op,)
            # ── SAM 侧处理（HBM block） ──
            if edit.type == "prefetch":
                get_block_info_fn = self._get_block_hashes_by_session
                process_session_block_info = get_block_info_fn(session_id, edit.block_start, edit.block_end)
                process_num = sum(len(info) for info in process_session_block_info)
                logger.info(f"session {session_id} hash count {process_num}")
                actual_process_blocks = fn(session_id, process_session_block_info, is_session_op)
            else:
                get_block_info_fn = self._get_global_block_id_by_session
                process_session_block_info = get_block_info_fn(session_id, edit.block_start, edit.block_end)
                process_num = sum(len(info) for info in process_session_block_info)
                logger.info(f"session {session_id} block id count {process_num}")
                actual_process_blocks = fn(session_id, edit.block_start, edit.block_end, process_session_block_info,
                                           is_session_op)

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
    """单个 block hash 的完整状态 — 被引用状态 + offload 阶段计数。

    SSM 不感知 hash 的具体缓存位置，只维护一个 offload 阶段计数：
    - 0：未 offload（HBM）
    - 1：已 offload 一级
    - 2：已下沉到底，达到上限，不能再 offload（进入此阶段时通知 SPM 释放）
    """
    block_hash: BlockHash
    ref_count: int = 0
    ref_sessions: set[str] = field(default_factory=set)
    offload_count: int = 0
    # offload 阶段上限。0→1 只记录状态；1→2 才触发 on_offload 回调通知 SPM。
    max_offload_count: int = 2

    # ── 引用计数操作 ──────────────────────────────────────────────

    def add_ref(self, session_id: str) -> bool:
        """增加一个 session 引用。返回是否发生了 offload 状态复位。"""
        self.ref_count += 1
        self.ref_sessions.add(session_id)
        if self.offload_count != 0:
            self.offload_count = 0
            return True  # was reset from offloaded state back to HBM
        return False  # already in HBM (offload_count == 0)

    def remove_ref(self, session_id: str) -> bool:
        """减少一个 session 引用。ref 归零时即完全释放。返回是否已完全释放。"""
        self.ref_count -= 1
        self.ref_sessions.discard(session_id)
        return self.ref_count == 0

    # ── offload 阶段操作 ──────────────────────────────────────────

    def offload_once(self) -> bool:
        """offload 一级。返回是否需要通知 SPM（恰从 1 进入 2）。

        0 → 1：仅内部状态推进，不通知 SPM（数据还在本节点缓存）。
        1 → 2：下沉到底，此时才需要通知 SPM 真正释放远端数据。
        已是 2（>= max）：不能再 offload，返回 False。
        """
        if self.offload_count >= self.max_offload_count:
            return False
        self.offload_count += 1
        return self.offload_count == self.max_offload_count

    # ── 查询 ──────────────────────────────────────────────────────

    def is_shared_with_active_session(self, requesting_session: str, is_active_fn) -> bool:
        """判断 hash 是否被其他活跃 session 共享（多 session 保护检查）。"""
        if self.ref_count <= 1:
            return False
        other_sessions = self.ref_sessions - {requesting_session}
        return any(is_active_fn(sid) for sid in other_sessions)

@dataclass
class EvictedSessionInfo:
    """LRU 淘汰结果"""
    session_id: str
    freed_hashes: list[BlockHash]
    still_shared_hashes: list[BlockHash]

class SessionStateManager:
    """Manages block hash state, offload phases, and LRU eviction.

    SSM 不感知 hash 的具体缓存位置，只维护每个 hash 的引用计数与
    offload 阶段计数（见 BlockHashState）。具体缓存分层由上层 SPM 负责。
    """

    def __init__(
        self,
        max_sessions: int = 1000,
        on_offload: Callable[[list[BlockHash]], None] | None = None,
        on_evict: Callable[[list[BlockHash]], int | None] | None = None,
    ):
        self._session_block_hash: dict[str, list[BlockHash]] = {}
        self._global_hash_state: dict[BlockHash, BlockHashState] = {}
        self._lru: OrderedDict[str, float] = OrderedDict()
        self._session_hash_count: dict[str, int] = {}
        self._active_session_ids: set[str] = set()  # Sessions with active requests
        self.max_sessions = max_sessions

        # SPM callbacks for context management notifications
        self._on_offload = on_offload
        self._on_evict = on_evict

    # ════════════════════════════════════════════════════════════════
    # Active session tracking (notified by SAM on request alloc/free)
    # ════════════════════════════════════════════════════════════════

    def mark_session_active(self, session_id: str) -> None:
        """Mark session as having active requests."""
        if session_id not in self._active_session_ids:
            logger.info("SSM: mark_session_active [%s]", session_id)
            self._active_session_ids.add(session_id)

    def mark_session_inactive(self, session_id: str) -> None:
        """Mark session as having no active requests."""
        if session_id in self._active_session_ids:
            logger.info("SSM: mark_session_inactive [%s]", session_id)
            self._active_session_ids.discard(session_id)

    def _is_session_active(self, session_id: str) -> bool:
        """Check if the session currently has active requests."""
        return session_id in self._active_session_ids

    def set_session_block_hashes(
        self, session_id: str, new_hashes: list[BlockHash],
    ) -> None:
        """Overwrite the session's block hash list (no diff). Session is guaranteed active at this point."""
        old_hashes = self._session_block_hash.get(session_id, [])
        if old_hashes == new_hashes:
            logger.debug("SSM: set_session_block_hashes [%s] unchanged %d hashes, skipped",
                          session_id, len(old_hashes))
            self.mark_session_active(session_id)
            self._touch_lru(session_id)
            return

        logger.info("SSM: set_session_block_hashes [%s] %d -> %d hashes",
                     session_id, len(old_hashes), len(new_hashes))

        for h in old_hashes:
            self._unregister_hash(h, session_id)

        # Register new hashes
        new_count = 0
        reset_count = 0
        for h in new_hashes:
            if h in self._global_hash_state:
                if self._global_hash_state[h].add_ref(session_id):
                    reset_count += 1
            else:
                self._global_hash_state[h] = BlockHashState(
                    block_hash=h, ref_count=1, ref_sessions={session_id},
                    offload_count=0,
                )
                new_count += 1

        self._session_block_hash[session_id] = new_hashes
        self._session_hash_count[session_id] = len(new_hashes)
        logger.info("SSM: set_session_block_hashes [%s] %d new global entries, %d reset to HBM, total managed hashes=%d",
                      session_id, new_count, reset_count, len(self._global_hash_state))
        self.mark_session_active(session_id)
        self._touch_lru(session_id)

    def _unregister_hash(self, block_hash: BlockHash, session_id: str) -> bool:
        """Unregister a session's reference to a hash. Returns True if the hash was fully freed."""
        state = self._global_hash_state.get(block_hash)
        if state is None:
            return False
        if state.remove_ref(session_id):
            del self._global_hash_state[block_hash]
            return True
        return False

    def _batch_offload_once(
        self, block_hashes: list[BlockHash]
    ) -> list[BlockHash]:
        """对每个 hash 推进一次 offload 阶段，返回需要通知 SPM 的 hash。

        新语义：
        - offload_count 0→1：仅内部状态推进，不回调；
        - offload_count 1→2：下沉到底，需调用 on_offload 通知 SPM；
        - 已是 2：不能再 offload，跳过。
        """
        to_notify: list[BlockHash] = []
        for block_hash in block_hashes:
            state = self._global_hash_state.get(block_hash)
            if state is None:
                continue
            if state.offload_once():
                to_notify.append(block_hash)
        logger.info("SSM: _batch_offload_once %d hashes updated, %d need SPM notify",
                    len(block_hashes), len(to_notify))
        return to_notify

    def _get_hashes_with_offload(
        self, session_id: str, start: int, end: int,
    ) -> list[tuple[BlockHash, int]]:
        """Return [(hash, offload_count)] for the session's [start, end) range.

        Hashes without a global state fall back to offload_count=0 (未 offload)。
        """
        hashes = self._session_block_hash.get(session_id, [])
        if not hashes:
            return []
        if end == -1:
            end = len(hashes)
        start = max(0, start)
        end = min(end, len(hashes))
        if start >= end:
            return []
        result = []
        for h in hashes[start:end]:
            state = self._global_hash_state.get(h)
            result.append((h, state.offload_count if state is not None else 0))
        return result

    def _filter_hashes_with_active_share(
        self, hashes: list[BlockHash], requesting_session: str,
    ) -> list[BlockHash]:
        filtered = []
        for h in hashes:
            state = self._global_hash_state.get(h)
            if state is None or not state.is_shared_with_active_session(requesting_session, self._is_session_active):
                filtered.append(h)
        return filtered

    def process_offload_hashes(
        self, session_id: str, block_start: int, block_end: int,
        **kwargs,
    ) -> None:
        """SSM-side offload: advance each hash's offload phase, notify SPM.

        新语义（SSM 不感知具体缓存位置，只推进 offload_count 阶段）：
        - 0 → 1：仅内部状态推进，不调用 _on_offload；
        - 1 → 2：下沉到底（达到 max_offload_count），需要调用 _on_offload
          通知 SPM 真正释放远端数据；
        - 2 及以上：已达上限，不能再 offload，跳过。
        被其他活跃 session 共享的 hash 依旧跳过（多 session 保护）。
        """
        hashes_with_offload = self._get_hashes_with_offload(
            session_id, block_start, block_end)
        if not hashes_with_offload:
            logger.info("SSM: process_offload_hashes [%s] no hashes in [%d, %d)", session_id, block_start, block_end)
            return
        logger.info("SSM: process_offload_hashes [%s] %d hashes in [%d, %d)",
                     session_id, len(hashes_with_offload), block_start, block_end)

        offloadable: list[BlockHash] = []
        capped_skipped = 0
        for h, cnt in hashes_with_offload:
            state = self._global_hash_state.get(h)
            if state is None:
                continue
            # Multi-session protection: skip hashes referenced by other active sessions
            if state.is_shared_with_active_session(session_id, self._is_session_active):
                continue
            if cnt >= state.max_offload_count:
                capped_skipped += 1
                continue
            offloadable.append(h)

        if not offloadable:
            logger.info("SSM: process_offload_hashes [%s] no offloadable hashes%s",
                         session_id,
                         f", {capped_skipped} already capped" if capped_skipped else "")
            return

        # 只推进阶段并返回恰进入底层的 hash（1→2）用于通知 SPM。
        to_notify = self._batch_offload_once(offloadable)

        msg = f"SSM: process_offload_hashes [{session_id}] {len(offloadable)} hashes advanced"
        if to_notify:
            msg += f", {len(to_notify)} reached bottom (notify SPM)"
        if capped_skipped:
            msg += f", {capped_skipped} already capped"
        logger.info(msg)
        self._touch_lru(session_id)
        if self._on_offload is not None and to_notify:
            self._on_offload(to_notify)

    def process_evict_hashes(
        self, session_id: str, block_start: int = 0, block_end: int = -1,
        is_session: bool = False,
    ) -> None:
        """SSM-side evict: unregister hash references, notify SPM.

        - Session evict: unregister all hashes and clean up LRU state
        - Range evict: unregister hashes in [block_start, block_end)
        Callback only receives hashes that were fully freed (ref_count reached 0).
        """
        if is_session:
            self._process_session_evict(session_id)
        else:
            self._process_range_evict(session_id, block_start, block_end)

    def _process_session_evict(self, session_id: str) -> None:
        """Evict all hashes for a session and remove from LRU. Only freed hashes in callback."""
        if not self._is_session_managed(session_id):
            logger.info("SSM: process_evict_hashes [%s] session evict skipped, already evicted from LRU", session_id)
            return
        logger.info("SSM: process_evict_hashes [%s] session evict", session_id)
        results = self._evict_lru_sessions([session_id])
        if self._on_evict is not None and results:
            self._on_evict(results[0].freed_hashes)

    def _process_range_evict(
        self, session_id: str, block_start: int, block_end: int,
    ) -> None:
        """Evict a range of hashes for a session. Only freed hashes in callback."""
        hashes_with_offload = self._get_hashes_with_offload(
            session_id, block_start, block_end)
        if not hashes_with_offload:
            logger.warning("SSM: process_evict_hashes [%s] no hashes in [%d, %d)",
                         session_id, block_start, block_end)
            return

        target_hashes = [h for h, _ in hashes_with_offload]
        safe_hashes = self._filter_hashes_with_active_share(target_hashes, session_id)
        shared_count = len(target_hashes) - len(safe_hashes)

        if not safe_hashes:
            logger.warning("SSM: process_evict_hashes [%s] all %d hashes protected by active sessions",
                         session_id, len(target_hashes))
            return

        # Unregister and separate freed vs. still-shared
        freed_hashes: list[BlockHash] = []
        still_shared_count = 0
        for h in safe_hashes:
            if self._unregister_hash(h, session_id):
                freed_hashes.append(h)
            else:
                still_shared_count += 1

        logger.info("SSM: process_evict_hashes [%s] %d freed, %d still shared (filtered from %d target, %d active-share protected)",
                     session_id, len(freed_hashes), still_shared_count,
                     len(target_hashes), shared_count)

        # Touch LRU if session still has remaining hashes
        if session_id in self._session_block_hash and self._session_block_hash[session_id]:
            self._touch_lru(session_id)

        if self._on_evict is not None and freed_hashes:
            self._on_evict(freed_hashes)

    def touch_lru(self, session_id: str) -> None:
        """Move session to the end of the LRU list (most recently used)."""
        if session_id in self._lru:
            self._lru.move_to_end(session_id)
        else:
            self._lru[session_id] = time.monotonic()
            logger.info("SSM: touch_lru [%s] newly tracked in LRU (total=%d)",
                         session_id, len(self._lru))

    def _touch_lru(self, session_id: str) -> None:
        self.touch_lru(session_id)

    def process_prefetch_hashes(
        self, session_id: str, block_hashes: list[BlockHash],
    ) -> None:
        """SSM-side prefetch: reset hash offload phase back to HBM (count=0).

        SPM 在真正预取后调用。对每个 hash 将 offload_count 重置为 0
        （表示回到 HBM）；如果 session 在 SSM 中无记录（被 LRU 淘汰过），
        自动重建。
        """
        session_known = session_id in self._session_block_hash

        for h in block_hashes:
            state = self._global_hash_state.get(h)
            if state is not None:
                state.offload_count = 0  # reset to HBM phase
            else:
                self._global_hash_state[h] = BlockHashState(
                    block_hash=h, ref_count=1, ref_sessions={session_id},
                    offload_count=0,
                )

        if not session_known:
            self._session_block_hash[session_id] = block_hashes.copy()
            self._session_hash_count[session_id] = len(block_hashes)
            logger.info("SSM: process_prefetch_hashes [%s] new session with %d hashes",
                         session_id, len(block_hashes))
        else:
            logger.info("SSM: process_prefetch_hashes [%s] reset %d hashes to HBM",
                         session_id, len(block_hashes))
        self._touch_lru(session_id)

    def _get_evict_candidates(
        self, count: int = 1, active_session_ids: set[str] | None = None,
    ) -> list[str]:
        active = active_session_ids or set()
        # 从 inactive 中按 hash_count 升序 + LRU 年龄排序
        # _lru 迭代顺序：越靠前越老（最早 touch 或最早添加）
        candidates = [sid for sid in self._lru if sid not in active]
        # 相同 hash_count 时 stable sort 保持 _lru 顺序（老的在前）
        candidates.sort(key=lambda sid: self._session_hash_count.get(sid, 0))
        return candidates[:count]

    def _try_evict_if_needed(self) -> None:
        if len(self._lru) < self.max_sessions:
            return
        candidates = self._get_evict_candidates(count=1)
        if candidates:
            logger.info("SSM: _try_evict_if_needed LRU full (%d >= %d), evicting [%s]",
                         len(self._lru), self.max_sessions, candidates[0])
            self._evict_lru_sessions(session_ids=candidates)

    def _evict_lru_sessions(self, session_ids: list[str]) -> list[EvictedSessionInfo]:
        results = []
        for session_id in session_ids:
            if session_id not in self._lru:
                continue
            hashes = self._session_block_hash.get(session_id, [])
            freed_hashes: list[BlockHash] = []
            still_shared_hashes: list[BlockHash] = []
            for h in hashes:
                state = self._global_hash_state.get(h)
                if state is None:
                    continue
                if state.remove_ref(session_id):
                    del self._global_hash_state[h]
                    freed_hashes.append(h)
                else:
                    still_shared_hashes.append(h)
            self._session_block_hash.pop(session_id, None)
            self._session_hash_count.pop(session_id, None)
            self._lru.pop(session_id, None)
            results.append(EvictedSessionInfo(
                session_id=session_id, freed_hashes=freed_hashes,
                still_shared_hashes=still_shared_hashes,
            ))
            logger.info("SSM: evicted session [%s] (%d hashes: %d freed, %d still shared)",
                         session_id, len(hashes), len(freed_hashes), len(still_shared_hashes))
        return results

    def _is_session_managed(self, session_id: str) -> bool:
        return session_id in self._lru

