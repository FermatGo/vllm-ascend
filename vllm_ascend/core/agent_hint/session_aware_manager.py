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
    """Session-Aware Manager — HBM 物理资源 + TTL 的权威维护者。

    数据结构（设计 §2.2.3）：两张权威表 + block 字段。

    - ``_session_hashes: dict[sid, list[BlockHashState]]`` — session → 按
      hash_block_size 粒度保序的 BlockHashState 列表。与 SSM 共享同一份
      dict 引用：SAM 只读 / 切片（register 时由 SSM 写入初始 list，
      evict 时由 SSM trim / pop）。
    - ``_hash_state: dict[BlockHash, BlockHashState]`` — 基础 hash 全局反查。
      与 SSM 共享同一份 dict 引用：SAM 只读（allocate/hit 时填
      ``block_ids_per_group``），SSM 负责增删 entry 与字段维护。
    - ``block.sessions: set[str]`` — block → session 反向索引（替代原
      ``_block_sessions`` 3D 表），由本类在 allocate / cache-hit 路径写入，
      在 evict / hash 复用清理时 discard / clear。

    不再维护：``_sessions``（父子表，父子语义由上层维护）、
    ``_session_blocks`` / ``_block_sessions``（双向索引表）、
    ``_session_block_hash``（原始 hash 列表表）。
    """

    def __init__(self, kv_cache_manager: KVCacheManager):
        logger.info(
            "SAM: __init__ enter num_kv_cache_groups=%s block_size=%s hash_block_size=%s",
            kv_cache_manager.num_kv_cache_groups,
            tuple(sprc.block_size for sprc in kv_cache_manager.coordinator.single_type_managers),
            kv_cache_manager.block_pool.hash_block_size,
        )
        self.kv_cache_manager = kv_cache_manager
        self.num_kv_cache_groups = kv_cache_manager.num_kv_cache_groups
        self.block_size = tuple(sprc.block_size for sprc in self.kv_cache_manager.coordinator.single_type_managers)
        self.hash_block_size = self.kv_cache_manager.block_pool.hash_block_size

        # 两张权威表（与 SSM 共享同一份 dict 引用）
        self._session_hashes: dict[str, list[BlockHashState]] = {}
        self._hash_state: dict[BlockHash, BlockHashState] = {}

        # TTL 管理器（定时轮）
        self._ttl_manager = TTLManager(on_expired=self._on_ttl_expired, notify_func=self._notify_event)

        # 事件监听器列表
        self._event_listeners: list[SessionEventListener] = []

        self._session_state_manager = SessionStateManager(
            max_sessions=100,
            session_hash_state=self._session_hashes,
            hash_state=self._hash_state,
            apply_evict_remote_cache_fn=self._on_session_evicted_remote,
            apply_evict_kvblock_fn=self.apply_evict_kvblock,
        )

        # Session 控制器（三层编排：SAM 切片 → SSM 分类+改 state → SAM apply + SPM 远端）
        self._session_controller = SessionController(
            get_hash_states_for_range=self.get_session_hash_states_for_range,
            get_block_hashes_by_session=self._get_session_block_hash,
            execute_prefetch=self._execute_prefetch,
            apply_offload_kvblock=self.apply_offload_kvblock,
            apply_evict_kvblock=self.apply_evict_kvblock,
            ssm_classify_offload=self._session_state_manager.classify_offload_work,
            ssm_classify_evict=self._session_state_manager.classify_evict_work,
            spm_notify=self._notify_event,
        )
        logger.debug("SAM: __init__ done ttl_manager=%s ssm=%s controller=%s",
                     self._ttl_manager is not None,
                     self._session_state_manager is not None,
                     self._session_controller is not None)

    def on_blocks_allocated_for_request(
        self, request: Request, blocks: KVCacheBlocks, cached_blocks_len_before: tuple[int, ...] | None = None
    ) -> None:
        # 解析引擎层透传的 opaque dict 为 ascend 强类型视图
        # 结果已缓存到 request._agent_hint_params，避免每个 block 都重新解析。
        hint = get_agent_hint_params(request)
        # 如果 request 没有 session_id，则不进行 session 相关的处理
        # 但是清理所有 block 的 session 引用和 TTL，避免残留
        if not hint or hint.session_id is None:
            total_blocks = sum(len(bids) for bids in blocks.get_block_ids())
            logger.debug(
                "SAM: on_blocks_allocated_for_request no session, request=%s total_blocks=%d (clearing refs)",
                request.request_id, total_blocks,
            )
            for group_id, block_ids in enumerate(blocks.get_block_ids()):
                for block_id in block_ids:
                    block = self._get_block(block_id)

                    # 清理该物理 block 的所有旧 session 引用（含 TTL 与 state 反向清理）
                    self.clear_block_session_refs(block_id)

                    # block本身清理的session_ref_cnt和ttl_expire_at
                    self._reset_block_state(block)
            return

        is_prefill = request.num_output_tokens == 0
        per_group_counts = [len(bids) for bids in blocks.get_block_ids()]
        logger.info(
            f"SAM: on_blocks_allocated_for_request request={request.request_id} "
            f"session={hint.session_id} is_prefill={is_prefill} "
            f"per_group_blocks={per_group_counts} cached_before={cached_blocks_len_before}"
        )

        self._register_blocks_to_session(
            session_id=hint.session_id,
            blocks=blocks,
            ephemeral_range=compute_ephemeral_range(hint.cache_control),
            cached_blocks_len_before=cached_blocks_len_before,
            is_prefill=is_prefill,
            is_cache_hit=False,
        )

    def on_block_cache_hit_for_request(self, request: Request, blocks: KVCacheBlocks) -> None:
        hint = get_agent_hint_params(request)
        if hint is None or hint.session_id is None:
            logger.debug("SAM: on_block_cache_hit_for_request no hint/session, request=%s skipped", request.request_id)
            return

        per_group_counts = [len(bids) for bids in blocks.get_block_ids()]
        logger.info(
            f"SAM: on_block_cache_hit_for_request request={request.request_id} "
            f"session={hint.session_id} per_group_blocks={per_group_counts}"
        )

        self._register_blocks_to_session(
            session_id=hint.session_id,
            blocks=blocks,
            ephemeral_range=compute_ephemeral_range(hint.cache_control),
            cached_blocks_len_before=None,
            is_prefill=False,
            is_cache_hit=True,
        )

    def _register_blocks_to_session(
        self,
        session_id: str,
        blocks: KVCacheBlocks,
        ephemeral_range: EphemeralRange | None,
        cached_blocks_len_before: tuple[int, ...] | None,
        is_prefill: bool,
        is_cache_hit: bool,
    ) -> None:
        """allocate / cache-hit 统一注册路径（设计 §4.1）。

        两个路径都要写 ``block.sessions`` + 填 ``BlockHashState.block_ids_per_group``。
        state 层的 ref_count / ref_sessions 已由 register_session_block_hash
        （on_request_scheduled 时 SSM diff 注册）建立，本方法不重复加 ref，
        只维护 HBM 物理引用与 block meta：

        - allocate（新分配 block，block.block_hash 为本轮新写入）：防御性
          清理残留引用（正常情况下 block 复用钩子已在 hash 重置前清理过），
          重置 block session 状态后建立当前 session 引用；
        - cache hit（prefix cache 命中复用 block）：block 上其他 session 的
          引用有效，不清理；当前 session 未注册则加引用，已注册则按需
          刷新 ephemeral TTL（取更晚的过期时间）。
        """
        now = time.monotonic()
        newly_protected_hashes: list[tuple(int, list[BlockHash])] = []
        newly_protected_ttl: float = 0.0
        protected_block_hashes: list[BlockHash] = []

        logger.debug(
            "SAM: _register_blocks_to_session enter session=%s is_cache_hit=%s is_prefill=%s "
            "ephemeral=%s cached_before=%s",
            session_id, is_cache_hit, is_prefill,
            ephemeral_range is not None, cached_blocks_len_before,
        )

        for group_id, block_ids in enumerate(blocks.get_block_ids()):
            logger.debug(f"SAM: _register_blocks_to_session group=%d block_ids={block_ids}", group_id)
            cached_blocks_len = cached_blocks_len_before[group_id] if cached_blocks_len_before else 0

            for ind, block_id in enumerate(block_ids):
                block = self._get_block(block_id)

                if not is_cache_hit:
                    # 防御性清理：正常流程中 block 复用钩子（hash 重置前）
                    # 已清掉旧 session 引用；此处兜底处理绕过钩子的路径。
                    # 必须在 reset_session_state 之前调用（需要 block_hash
                    # 反查 _hash_state）。新分配 block 属于活跃请求
                    # （ref_cnt > 0，不在 free queue），重置安全。
                    self.clear_block_session_refs(block_id)
                    self._reset_block_state(block)

                # allocate_slots理论上只传 newly-cached blocks，保留检查用于防御异常情况。
                if block.block_hash is None:
                    logger.debug("SAM: _register_blocks_to_session block=%d has no block_hash, skip", block_id)
                    continue

                physical_hash = get_block_hash(block.block_hash)
                state = self._hash_state.get(physical_hash)

                # 填 BlockHashState.block_ids_per_group（两个路径都填；
                # register 时为全 None，此处填入实际 block_id）
                if state is not None:
                    state.add_block_id(group_id, block_id)
                else:
                    logger.debug(
                        "SAM: _register_blocks_to_session block=%d hash=%s not in _hash_state (not registered yet?)",
                        block_id, physical_hash,
                    )

                # ephemeral TTL 判定（沿用现有逻辑）
                if is_cache_hit:
                    is_ephemeral = (
                        ephemeral_range is not None
                        and ephemeral_range.ttl > 0
                        and self.block_size[group_id] // self.hash_block_size * ind <= ephemeral_range.block_offset
                    )
                else:
                    is_ephemeral = (
                        ephemeral_range is not None
                        and ephemeral_range.ttl > 0
                        and self.block_size[group_id] // self.hash_block_size * (cached_blocks_len + ind)
                        <= ephemeral_range.block_offset
                        and is_prefill
                    )
                ttl_expire_at = now + ephemeral_range.ttl if is_ephemeral and ephemeral_range is not None else 0.0

                if not is_cache_hit:
                    # 新分配：建立当前 session 引用（block 已在前面
                    # clear + reset 过，sessions 为空，add 即为唯一引用）
                    block.sessions.add(session_id)
                    self._update_block_meta(
                        block_id,
                        ttl_expire_at=ttl_expire_at,
                    )
                    if is_ephemeral:
                        newly_protected_hashes.append((block_id, block.block_hash))
                        newly_protected_ttl = ttl_expire_at
                        logger.debug(
                            "SAM: _register_blocks_to_session block=%d marked ephemeral ttl=%.2f",
                            block_id, ttl_expire_at,
                        )
                else:
                    # cache hit：是否已注册改为 sid in block.sessions
                    is_registered = session_id in block.sessions
                    if not is_registered:
                        block.sessions.add(session_id)
                        self._update_block_meta(
                            block_id,
                            # None 表示不覆盖其他 session 已经设置的 block TTL。
                            ttl_expire_at=ttl_expire_at if is_ephemeral else None,
                        )
                        if is_ephemeral:
                            newly_protected_hashes.append((block_id, block.block_hash))
                            newly_protected_ttl = ttl_expire_at
                            logger.debug(
                                "SAM: _register_blocks_to_session cache-hit block=%d new ephemeral ttl=%.2f",
                                block_id, ttl_expire_at,
                            )
                    elif is_ephemeral:
                        # 已注册则刷新 TTL：block 级取更晚的过期时间。
                        new_expire_at = max(block.ttl_expire_at, ttl_expire_at)
                        self._update_block_meta(
                            block_id,
                            ttl_expire_at=new_expire_at,
                        )
                        newly_protected_hashes.append((block_id, block.block_hash))
                        newly_protected_ttl = new_expire_at
                        logger.debug(
                            "SAM: _register_blocks_to_session cache-hit block=%d refreshed ephemeral ttl=%.2f (was %.2f)",
                            block_id, new_expire_at, block.ttl_expire_at,
                        )

        logger.debug(
            "SAM: _register_blocks_to_session exit session=%s newly_protected=%d protected_block_hashes=%d "
            "ttl_to_register=%.2f",
            session_id, len(newly_protected_hashes), len(protected_block_hashes), newly_protected_ttl,
        )
        if ephemeral_range:
            scale_factor = max(self.block_size) // self.hash_block_size
            session_states = self._session_hashes.get(session_id, [])
            block_end = (ephemeral_range.block_offset + scale_factor) // scale_factor * scale_factor
            if ephemeral_range.block_offset + 1 > len(session_states):
                logger.warning(f"SAM: ephemeral range exceed session block hash. session id:{session_id}")
                block_end = len(session_states)

            if newly_protected_hashes and session_states:
                protected_block_hashes = [s.block_hash for s in session_states[0:block_end]]

        # 仅在有真实保护内容时才注册，避免无效调用 + 早退噪音
        if protected_block_hashes and newly_protected_ttl > 0.0:
            self._ttl_manager.register(
                block_infos=[],
                session_id=session_id,
                expire_at=newly_protected_ttl,
                protected_block_hashes=protected_block_hashes,
            )
        else:
            logger.debug(
                "SAM: _register_blocks_to_session skip TTL register (protected=%d ttl=%.2f)",
                len(protected_block_hashes), newly_protected_ttl,
            )

    def _on_ttl_expired(self, block_id: int, session_id: str) -> None:
        """ephemeral block TTL 到期回调"""
        logger.debug(
            "SAM: _on_ttl_expired block=%d session=%s — clearing block TTL (session refs preserved)",
            block_id, session_id,
        )

        # 统一接口：TTL 到期只清除 block 的 TTL，不动两个引用计数
        # （session 引用仍在，num_session_refs 保持为 block.sessions
        # 的投影）。这会导致 block 的 is_ephemeral() 返回 False，
        # FreeKVCacheBlockQueue.on_block_meta_changed 自动重评估分区，
        # 剩余引用使 block 落入 B 区。
        self._update_block_meta(
            block_id,
            ttl_expire_at=0.0,
        )

    def clear_block_session_refs(self, block_id: int) -> None:
        """清除指定 block 的所有 session 引用（§7.1：SAM 只改 HBM block + TTL）。

        只清理 block 侧：``block.sessions.clear()`` + 两个计数归零
        （``set_session_state``）+ 清理各旧 session 在 TTLManager 中对
        此 block 的 per-block 表项。**不改 state 层**（ref_count /
        ref_sessions / ``_hash_state`` / ``_session_hashes`` 由 SSM 维护）——
        被复用 block 上旧 session 的 state 引用由 SSM 既有路径收敛：
        该 session 下次请求的 register diff（removed 分支减 ref）、
        显式 evict / LRU 驱逐（classify_evict_work 减 ref）。

        反查 ``_hash_state`` 仅只读（拿 block_ids_per_group 做调试日志），
        与 §7.1 读路径约定一致。
        """
        block = self._get_block(block_id)
        session_ids = list(block.sessions)
        if not session_ids:
            logger.debug("SAM: clear_block_session_refs block=%d no sessions, no-op", block_id)
            return

        # 只读反查：日志观察当前 hash 维度绑定（不修改 state）
        if block.block_hash is not None:
            state = self._hash_state.get(get_block_hash(block.block_hash))
            if state is not None:
                logger.debug(
                    "SAM: clear_block_session_refs block=%d hash_state=%s block_ids_per_group=%s clearing %d sessions=%s",
                    block_id, state.block_hash, state.block_ids_per_group, len(session_ids), session_ids,
                )
            else:
                logger.debug(
                    "SAM: clear_block_session_refs block=%d hash=%s not in _hash_state, clearing %d sessions",
                    block_id, get_block_hash(block.block_hash), len(session_ids),
                )
        else:
            logger.debug(
                "SAM: clear_block_session_refs block=%d no block_hash, clearing %d sessions",
                block_id, len(session_ids),
            )

        block.sessions.clear()
        # 物理引用全部终结（num_session_refs 派生自 sessions，自动归零）
        self._update_block_meta(
            block_id,
            ttl_expire_at=0.0,
        )

    def get_session_hash_states_for_range(
        self,
        session_id: str,
        block_start: int,
        block_end: int,
    ) -> tuple[list[BlockHashState], int, int]:
        """expand + 切片 ``_session_hashes[sid]`` 返回 ``list[BlockHashState]``。

        返回 (states, expanded_start, expanded_end)：切片后的 state 列表及
        扩展后的区间（hash_block_size 粒度、左闭右开）。扩展后的区间供
        SSM ``classify_evict_work`` 做对应的 trim，保证 trim 范围与物理
        block 边界对齐（设计 §4.4 / §5）。
        """
        states = self._session_hashes.get(session_id)
        if not states:
            logger.debug(
                "SAM: get_session_hash_states_for_range session=%s no states, returning empty",
                session_id,
            )
            return [], 0, 0

        total = len(states)
        if block_end == -1:
            block_end = total
        block_end = min(block_end, total)

        if block_start >= block_end:
            logger.debug(
                "SAM: get_session_hash_states_for_range session=%s invalid range [%d,%d) (total=%d), empty",
                session_id, block_start, block_end, total,
            )
            return [], block_start, block_end

        expanded_start, expanded_end = self._expand_session_block_range(session_id, block_start, block_end)
        sliced = states[expanded_start:expanded_end]
        logger.debug(
            "SAM: get_session_hash_states_for_range session=%s req=[%d,%d) expanded=[%d,%d) total=%d returned=%d",
            session_id, block_start, block_end, expanded_start, expanded_end, total, len(sliced),
        )
        return sliced, expanded_start, expanded_end

    def _on_session_evicted_remote(self, session_id: str, freed_hashes: list[BlockHash]) -> None:
        """evict 后统一通知 SPM（Controller 路径与 SSM LRU 路径共用）。

        - ``session_blocks_removed``：SPM 侧 SessionKeyTracker 记账清理；
        - ``context_management_evict``：远端数据实际删除（按 backend 实现）。
        """
        if not freed_hashes:
            logger.debug("SAM: _on_session_evicted_remote session=%s no freed_hashes, no-op", session_id)
            return
        logger.debug(
            "SAM: _on_session_evicted_remote session=%s freed_hashes=%d",
            session_id, len(freed_hashes),
        )
        self._notify_event("session_blocks_removed", session_id=session_id, block_hashes=freed_hashes)
        self._notify_event("context_management_evict", block_hashes=freed_hashes)

    def _execute_prefetch(self, session_id: str, block_hashes: list[BlockHash]) -> int:
        """通知 SPM 创建远端预取任务。"""
        logger.debug(
            "SAM: _execute_prefetch session=%s block_hashes=%d",
            session_id, len(block_hashes),
        )
        self._notify_event(
            "context_management_prefetch",
            session_id=session_id,
            block_hashes=block_hashes,
        )
        return len(block_hashes)

    def apply_evict_kvblock(
        self,
        session_id: str,
        work: list[tuple[BlockHashState, list[int | None]]],
        is_session_op: bool,
    ) -> list[BlockHash]:
        """SAM 侧 HBM + TTL apply for evict（设计 §4.4 / §4.5）。

        只改 HBM block + TTL：``block.sessions.discard`` /
        ``set_session_state`` + ``_ttl_manager.remove``。
        不碰 ref_count / ref_sessions / _hash_state / _session_hashes
        （这些由 SSM 在 classify_evict_work 中维护）。

        入参 work 为 SSM 分类结果（hbm_only_work + hbm_and_remote_work），
        每个 entry 为 ``(state, block_ids_per_group 快照)``。SAM 只对
        ``state.ref_count == 0``（独占 hash）的 state 清空 block 引用并计入
        freed_hashes；对共享 hash 的 block 仅 discard 当前 sid（block 仍被
        其他 session 持有，保持其 TTL / 计数不被破坏）。

        返回 freed_hashes：仅含 ref_count 归 0（无其他 session 共享）的
        hash，调用方据此通知 SPM 删除远端数据。
        """
        logger.debug(
            "SAM: apply_evict_kvblock enter session=%s is_session_op=%s work=%d",
            session_id, is_session_op, len(work),
        )
        freed_hashes: list[BlockHash] = []
        # 同一物理 block 可能被同一 session 的多个基础 hash 覆盖
        # （group block_size > hash_block_size），去重保证每个 block 只处理一次。
        seen_block_ids: set[int] = set()
        skipped_no_ref = 0

        for state, block_ids in work:
            is_exclusive = state.ref_count == 0
            if is_exclusive:
                freed_hashes.append(state.block_hash)
            for bid in block_ids:
                if bid is None or bid in seen_block_ids:
                    continue
                seen_block_ids.add(bid)
                block = self._get_block(bid)
                if session_id not in block.sessions:
                    # 该 session 在此 block 上没有物理引用（重复 evict /
                    # 引用已在 block 复用时清理），幂等跳过
                    skipped_no_ref += 1
                    continue
                block.sessions.discard(session_id)
                # 逐 block 清理该 session 的 ephemeral TTL 表项，
                # 防止残留条目到期后误清新使用者的保护
                self._update_block_meta(
                    bid,
                    # TTL 仅在 block 上不再有任何 session 引用时清零；
                    # 还有其他 session 持有时保留（其 ephemeral 保护可能仍在）
                    ttl_expire_at=0.0 if not block.sessions else None,
                )

        # TTL 清理在 SAM 内部做（SAM 负责 TTL）。session-evict 与 range-evict
        # 均清理该 session 的 hash 维度 TTL 记录（block_id=-1 语义，
        # 见设计 §4.4 TODO 说明：按整条 session 粒度清理）。
        # 仅在该 session 确实注册过 TTL 时才 remove，避免无谓 warning 噪音。
        if (-1, session_id) in self._ttl_manager._entries:
            self._ttl_manager.remove(block_id=-1, session_id=session_id)
        else:
            logger.debug(
                "SAM: apply_evict_kvblock [%s] no hash-level TTL entry, skip ttl_manager.remove",
                session_id,
            )

        if freed_hashes:
            logger.info(
                "SAM: apply_evict_kvblock [%s] is_session_op=%s states=%d freed_hashes=%d blocks=%d skipped_no_ref=%d",
                session_id, is_session_op, len(work), len(freed_hashes), len(seen_block_ids), skipped_no_ref,
            )
        else:
            logger.debug(
                "SAM: apply_evict_kvblock [%s] no exclusive hash freed (states=%d blocks=%d skipped_no_ref=%d)",
                session_id, len(work), len(seen_block_ids), skipped_no_ref,
            )
        return freed_hashes

    def _get_session_global_block_ids(
        self,
        session_id: str,
        block_start: int,
        block_end: int,
    ) -> tuple[tuple[list[int], ...], list[BlockHash]]:
        """将基础 hash block 范围映射为各 KV cache group 的物理 block id（设计 §4.7，改写）。

        从 ``_session_hashes[sid]`` 切片 BlockHashState，取
        ``state.block_ids_per_group``；同一物理 block_id 可能被多个 state
        共享（group 的 block_size > hash_block_size），per-group 去重保序。

        返回 (per-group block id tuple, union hash list)。
        """
        result: tuple[list[int], ...] = tuple([] for _ in range(self.num_kv_cache_groups))

        states = self._session_hashes.get(session_id)
        if not states:
            logger.warning(f"SAM: _get_session_global_block_ids no hash for session_id:{session_id}.")
            return result, []

        total = len(states)
        if block_end == -1:
            block_end = total
        block_end = min(block_end, total)

        # 防止负数下标被 Python 当成从末尾索引。
        if block_start < 0 or block_end < 0 or block_start >= block_end:
            logger.warning(
                "SAM: _get_session_global_block_ids invalid range for session %s: [%s, %s)",
                session_id, block_start, block_end,
            )
            return result, []

        # —— 用 _expand_session_block_range 扩展边界对齐物理 block ——
        expanded_start, expanded_end = self._expand_session_block_range(session_id, block_start, block_end)
        union_states = states[expanded_start:expanded_end]
        union_hashes = [s.block_hash for s in union_states]

        # per-group block_ids（去重，同一物理 block_id 可能被多个 state 共享）
        for state in union_states:
            for gid, bid in enumerate(state.block_ids_per_group):
                if bid is not None and bid not in result[gid]:
                    result[gid].append(bid)

        logger.debug(
            "SAM: _get_session_global_block_ids session=%s range=[%d,%d) expanded=[%d,%d) "
            "per_group_counts=%s union_hashes=%d",
            session_id, block_start, block_end, expanded_start, expanded_end,
            [len(g) for g in result], len(union_hashes),
        )
        return result, union_hashes

    def _get_session_block_hash(self, session_id: str, block_start: int, block_end: int) -> list[BlockHash]:
        """``_session_hashes[sid]`` 直接取，返回 ``[s.block_hash for s in states[start:end]]``。"""
        states = self._session_hashes.get(session_id)
        if not states:
            logger.debug("SAM: _get_session_block_hash session=%s no states, returning []", session_id)
            return []
        if block_end == -1:
            block_end = len(states)
        result = [s.block_hash for s in states[block_start:block_end]]
        logger.debug(
            "SAM: _get_session_block_hash session=%s range=[%d,%d) total=%d returned=%d",
            session_id, block_start, block_end, len(states), len(result),
        )
        return result

    def add_event_listener(self, listener: SessionEventListener):
        """注册事件监听器（SPM 调用）"""
        logger.debug("SAM: add_event_listener %r (total=%d)", listener, len(self._event_listeners) + 1)
        self._event_listeners.append(listener)

    def process_prefetch_req(self, num_unfinished_requests: int) -> None:
        """统一 prefetch 入口：调 SPM 处理预取，然后更新 SSM hash 位置为 HBM。"""
        logger.debug(
            "SAM: process_prefetch_req enter unfinished=%d listeners=%d",
            num_unfinished_requests, len(self._event_listeners),
        )
        for listener in tuple(self._event_listeners):
            handler = getattr(listener, "process_prefetch_req", None)
            if handler is not None:
                results = handler(num_unfinished_requests)
                n_results = len(results) if results else 0
                if n_results > 0:
                    logger.debug(
                        "SAM: process_prefetch_req listener=%r returned %d prefetch results",
                        listener, n_results,
                    )
                else:
                    logger.debug(
                        "SAM: process_prefetch_req listener=%r returned 0 prefetch results (no-op)",
                        listener,
                    )
                for session_id, block_hashes in results:
                    self._session_state_manager.process_prefetch_hashes(session_id, block_hashes)
                break
        else:
            logger.debug("SAM: process_prefetch_req no listener with process_prefetch_req handler")

    def _notify_event(self, event_type: str, **kwargs):
        """通知所有监听器"""
        ret = 0
        handler_count = 0
        for listener in tuple(self._event_listeners):
            handler = getattr(listener, f"on_{event_type}", None)
            if handler is None:
                continue
            handler_count += 1

            try:
                ret = handler(**kwargs)
            except Exception:
                logger.exception(
                    "SAM: _notify_event listener %r failed while handling %s kwargs=%s",
                    listener, event_type, list(kwargs.keys()),
                )
        logger.debug(
            "SAM: _notify_event type=%s listeners=%d handlers_invoked=%d",
            event_type, len(self._event_listeners), handler_count,
        )
        return ret

    def on_request_completed(self, request_id: str, session_id: str | None) -> None:
        """请求完成回调 — 处理挂起 edits 并通知 SSM 会话不再活跃"""
        logger.debug(
            "SAM: on_request_completed request=%s session=%s",
            request_id, session_id,
        )
        self._session_controller.on_request_completed(request_id)
        if session_id:
            self._session_state_manager.mark_session_inactive(session_id)
        else:
            logger.debug("SAM: on_request_completed request=%s no session_id, skip SSM inactive", request_id)

    def tick(self, now: float | None = None) -> None:
        """推进 TTL timer wheel，并在 LRU 满时尝试淘汰 session。由 scheduler 统一调用。"""
        logger.debug("SAM: tick enter now=%s", now)
        expired_before = len(self._ttl_manager._entries)
        self._ttl_manager.tick(now)
        expired_after = len(self._ttl_manager._entries)
        if expired_before != expired_after:
            logger.debug("SAM: tick expired %d TTL entries (now=%s)",
                        expired_before - expired_after, now)
        self._session_state_manager._try_evict_if_needed()

    def register_agent_hint(
        self,
        request_id: str,
        session_id: str | None,
        context_management: ContextManagementParams | None,
    ) -> list[Any] | None:
        """register context management to session controller"""
        has_edits = context_management is not None and context_management.edits is not None
        edit_count = len(context_management.edits) if has_edits else 0
        logger.debug(
            "SAM: register_agent_hint request=%s session=%s has_edits=%s edit_count=%d",
            request_id, session_id, has_edits, edit_count,
        )
        return self._session_controller.process_request_edits(request_id, session_id, context_management)

    def register_session_block_hash(self, session_id: str, block_hashes: list[BlockHash]) -> None:
        """register session block hash to session aware manager and session state manager.

        SSM 侧做 diff 注册（更新 _hash_state 的 ref_count/ref_sessions 并
        写入共享的 _session_hashes[sid]）；被移除 hash 的 state 层引用
        （减 ref / 归零删 entry）也由 SSM diff 完成。block 侧旧引用由
        block 复用清理路径（clear_block_session_refs）兜底收敛。
        block_ids_per_group 在 on_blocks_allocated / on_block_cache_hit
        回调中按物理 block 填入。
        """
        prev_hashes = self._session_hashes.get(session_id, [])
        prev_count = len(prev_hashes) if prev_hashes is not None else 0
        new_count = len(block_hashes)
        if new_count != prev_count:
            logger.debug(
                "SAM: register_session_block_hash session=%s block_hashes=%d → %d",
                session_id, prev_count, new_count,
            )
        else:
            logger.debug(
                "SAM: register_session_block_hash session=%s unchanged=%d (no-op)",
                session_id, new_count,
            )
        self._session_state_manager.set_session_block_hashes(session_id, block_hashes)

    def _get_block(self, block_id: int) -> KVCacheBlock:
        return self.kv_cache_manager.block_pool.blocks[block_id]

    def _reset_block_state(self, block: KVCacheBlock) -> None:
        """
        Reset the session state of a block, clearing its session
        reference count and TTL expiration time.
        """
        logger.debug("SAM: _reset_block_state block=%d", block.block_id if hasattr(block, 'block_id') else '?')
        block.reset_session_state()

    def _update_block_meta(
        self,
        block_id: int,
        ttl_expire_at: float | None = None,
    ) -> None:
        """Update block TTL metadata and re-evaluate the free queue zone.

        session 引用计数不再单独维护：``block.num_session_refs`` 由
        ``len(block.sessions)`` 实时派生（设计 §7.1 决策），调用方在
        改完 ``block.sessions`` 后调本方法触发分区重评估；B 区驻留
        完全由 sessions 非空决定（offload 不动 sessions，替代旧
        ``is_offload_block`` / 双计数机制）。``ttl_expire_at`` 为 None
        表示不覆盖现有 TTL。
        """
        block_pool = self.kv_cache_manager.block_pool
        block = block_pool.blocks[block_id]
        if ttl_expire_at is not None:
            block.set_session_state(ttl_expire_at=ttl_expire_at)
        if block.ref_cnt == 0 and not block.is_null:
            block_pool.free_block_queue.on_block_meta_changed(block)
            logger.debug(
                "SAM: _update_block_meta block=%d ttl=%s ref_cnt=0 → re-evaluated free queue zone",
                block_id, ttl_expire_at,
            )

    def _expand_session_block_range(
        self,
        session_id: str,
        block_start: int,
        block_end: int,
    ) -> tuple[int, int]:
        """扩展左闭右开区间，使其包含所有 group 涉及的完整 block（保留，设计 §8.1）。

        处理 group 的 block_size > hash_block_size 时逻辑范围与物理 block
        边界对齐的问题：states[start:end] 是 hash_block_size 粒度切片，
        但物理 block_id 跨多个基础 hash 共享，扩展范围保证切片覆盖完整
        物理 block。

        输入和输出均为 hash_block_size 粒度，区间语义均为
        [block_start, block_end)。
        """
        assert block_start >= 0
        assert block_end >= block_start

        # 空区间没有涉及任何物理 block。
        if block_start == block_end:
            logger.debug(
                "SAM: _expand_session_block_range session=%s empty range [%d,%d), no-op",
                session_id, block_start, block_end,
            )
            return block_start, block_end

        states = self._session_hashes.get(session_id)
        if not states:
            logger.warning(f"SAM: _expand_session_block_range no hash for session_id:{session_id}.")
            return 0, 0

        base_block_count = len(states)

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
        logger.debug(
            "SAM: _expand_session_block_range session=%s req=[%d,%d) → expanded=[%d,%d) (total=%d)",
            session_id, block_start, block_end, expanded_start, expanded_end, base_block_count,
        )
        return expanded_start, expanded_end

    def apply_offload_kvblock(
        self,
        session_id: str,
        hbm_work: list[tuple[BlockHashState, list[int | None]]],
    ) -> None:
        """SAM 侧 HBM apply for offload（设计 §4.3，实现说明见 §8.7 决议）。

        offload 不动 ``block.sessions`` / ``ref_cnt`` / ``ref_count``：
        session 仍逻辑持有 block，只是数据下沉。block 何时真正释放回
        free queue 仍由 vLLM 原生 ``block.ref_cnt`` 决定；由于
        ``len(block.sessions) > 0``，block 在 free queue 中自然留在 B 区
        （``is_offload_block`` 及其后的双计数方案均已由 sessions 集合
        取代：B 区驻留完全由 sessions 非空派生，无独立标记位）。

        因此 HBM 侧唯一动作是：若启用 ``_ENABLE_OFFLOAD_TTL_PROTECT``，
        注册 offload TTL 保护（1h hash 维度免疫期）。offload TTL 与
        ephemeral TTL 独立：前者保护 hash 维度，后者保护 block 维度。
        """
        logger.info(
            "SAM: apply_offload_kvblock enter session=%s hbm_work=%d ttl_protect=%s",
            session_id, len(hbm_work), _ENABLE_OFFLOAD_TTL_PROTECT,
        )
        if not hbm_work:
            logger.info("SAM: apply_offload_kvblock session=%s empty hbm_work, no-op", session_id)
            return
        if _ENABLE_OFFLOAD_TTL_PROTECT:
            self._ttl_manager.register(
                block_infos=[],
                session_id=session_id,
                expire_at=time.monotonic() + 3600,
                protected_block_hashes=[state.block_hash for state, _ in hbm_work],
            )
        logger.info(
            "SAM: apply_offload_kvblock [%s] first-offload states=%d (ttl_protect=%s)",
            session_id, len(hbm_work), _ENABLE_OFFLOAD_TTL_PROTECT,
        )


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
        block_id = block_id
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
    """Executor for ``context_management.edits`` — 三层编排者（设计 §3.3 / §7.3）。

    Processes the ``context_management.edits`` carried by a request,
    executing them immediately or deferring them until the request
    finishes based on the ``manage_request`` flag.

    Controller 只做编排，不做分类、不改 state、不直接改 block：
      1. 调 SAM 切片（expand + 切 ``_session_hashes``）拿 ``list[BlockHashState]``；
      2. 把 states 传给 SSM 分类 + 改 state（ref_count / ref_sessions /
         offload_count / 增删 _hash_state / trim _session_hashes）；
      3. 拿 SSM 返回的 work items 分发：HBM work 给 SAM（apply block+TTL），
         远端 work 给 SPM（通过事件通知）。

    Callbacks（构造时注册）:
    ----------------------------------------------------------------------
    SAM-side:
      ``get_hash_states_for_range``  (session_id, block_start, block_end)
          -> (list[BlockHashState], expanded_start, expanded_end)
      ``execute_prefetch``           (session_id, block_hashes) -> int
      ``apply_offload_kvblock``      (session_id, hbm_work) -> None
      ``apply_evict_kvblock``        (session_id, work, is_session_op) -> list[BlockHash]
          返回 freed_hashes（ref_count 归 0 的独占 hash），Controller 据此
          通知 SPM 删除远端数据。
    SSM-side:
      ``ssm_classify_offload``  (session_id, states) -> OffloadWork dict
      ``ssm_classify_evict``    (session_id, states, is_session_op, start, end) -> EvictWork dict
    Query:
      ``get_block_hashes_by_session`` (session_id, block_start, block_end) -> list[BlockHash]
          用于 block 范围合法性校验（total length）与 prefetch hash 解析。
    Event:
      ``spm_notify`` (event_type, **kwargs) -> None
          SAM 的事件分发入口（监听者为 SPM），用于 offload / evict 的远端动作。
    """

    def __init__(
        self,
        *,
        get_hash_states_for_range: Callable[..., tuple[list[BlockHashState], int, int]],
        get_block_hashes_by_session: Callable[[str, int, int], list[BlockHash]],
        execute_prefetch: Callable[..., int],
        apply_offload_kvblock: Callable[..., None],
        apply_evict_kvblock: Callable[..., list[BlockHash]],
        ssm_classify_offload: Callable[..., dict],
        ssm_classify_evict: Callable[..., dict],
        spm_notify: Callable[..., None],
    ):
        self._get_hash_states_for_range = get_hash_states_for_range
        self._get_block_hashes_by_session = get_block_hashes_by_session
        self._execute_prefetch = execute_prefetch
        self._apply_offload_kvblock = apply_offload_kvblock
        self._apply_evict_kvblock = apply_evict_kvblock
        self._ssm_classify_offload = ssm_classify_offload
        self._ssm_classify_evict = ssm_classify_evict
        self._spm_notify = spm_notify

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
        """执行单个 edit — 三层编排：SAM 切片 → SSM 分类+改 state → SAM apply + SPM 远端。"""
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

        if edit.type not in ("offload", "prefetch", "evict"):
            logger.warning("Unsupported edit type: %s, skipping.", edit.type)
            return EditResponse(
                session_id=session_id, type=edit.type, op_status=False, fail_reason=f"invalid edit type {edit.type}"
            )

        actual_process_blocks = 0
        op_result = True
        fail_reason = ""
        is_session_op = edit.target == "session"

        # 获取当前 session 基础 hash 总长度
        total_block_length = len(self._get_block_hashes_by_session(session_id, 0, -1))
        logger.info(f"Process session {session_id} with block hash length {total_block_length}")
        op_result, fail_reason = self.process_edit_index(edit, total_block_length, session_id)

        if op_result:
            if edit.type == "prefetch":
                # prefetch：无分类、无 state 改动，解析 hash 后直接通知 SPM
                block_hashes_for_spm = self._get_block_hashes_by_session(
                    session_id, edit.block_start, edit.block_end)
                actual_process_blocks = self._execute_prefetch(session_id, block_hashes_for_spm)
            else:
                # 1. SAM 切片：expand + 切 _session_hashes 拿 list[BlockHashState]
                #    （返回扩展后的区间，供 SSM trim 对齐物理 block 边界）
                states, expanded_start, expanded_end = self._get_hash_states_for_range(
                    session_id, edit.block_start, edit.block_end)

                if edit.type == "offload":
                    # 2. SSM 分类 + 改 state（active-share 过滤 + 推进 offload_count）
                    work = self._ssm_classify_offload(session_id, states)
                    # 3. Controller 分发：HBM work 给 SAM，远端 work 给 SPM
                    self._apply_offload_kvblock(session_id, work["hbm_work"])
                    if work["remote_work"]:
                        self._spm_notify("context_management_offload", block_hashes=work["remote_work"])
                    actual_process_blocks = len(states) - len(work["skipped"])
                else:  # evict（range evict / session evict 走同一三层流程）
                    # 2. SSM 分类 + 改 state（减 ref / del entry / trim 或 pop
                    #    _session_hashes，内部含 LRU 维护）
                    work = self._ssm_classify_evict(
                        session_id, states, is_session_op, expanded_start, expanded_end)
                    # 3. Controller 分发：SAM apply block+TTL；freed_hashes
                    #    （独占 hash）通知 SPM 删远端
                    freed_hashes = self._apply_evict_kvblock(
                        session_id,
                        work["hbm_only_work"] + work["hbm_and_remote_work"],
                        is_session_op,
                    )
                    if freed_hashes:
                        self._spm_notify(
                            "session_blocks_removed",
                            session_id=session_id,
                            block_hashes=freed_hashes,
                        )
                        self._spm_notify("context_management_evict", block_hashes=freed_hashes)
                    # 实际处理数 = 分类处理过的 state（含共享减 ref），
                    # skipped 不计；独占释放数另见 freed_hashes
                    actual_process_blocks = len(states) - len(work["skipped"])

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

    ``block_ids_per_group``: index = group_id, value = 物理 block_id；
    None 表示该 group 还没有对应的物理 block（hash 未填满该 group 的
    block）。同一物理 block_id 可能被多个 BlockHashState 共享（group 的
    block_size > hash_block_size 时，一个物理 block 覆盖多个基础 hash）。
    register 时初始化为全 None，由 SAM 在 allocate / cache-hit 回调中
    填入；SSM 在 classify 时快照并清空。SSM 不消费此字段做 ref/offload
    记账，仅供 SAM/SPM 定位物理 block。
    """
    block_hash: BlockHash
    ref_count: int = 0
    ref_sessions: set[str] = field(default_factory=set)
    offload_count: int = 0
    max_offload_count: int = 2
    # index = group_id, value = physical block_id (None = not materialized)
    block_ids_per_group: list[int | None] = field(default_factory=list)

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

    def add_block_id(self, group_id: int, block_id: int) -> None:
        """Register the physical block holding this hash in ``group_id``.

        Grows the list with ``None`` placeholders up to ``group_id`` first
        (register 时为空列表). Idempotent per group: the (hash, group)
        pair maps to exactly one cached block, repeated calls just
        overwrite with the same value. SAM calls this on alloc /
        cache-hit once the block↔hash mapping is established.
        """
        if group_id >= len(self.block_ids_per_group):
            self.block_ids_per_group.extend([None] * (group_id + 1 - len(self.block_ids_per_group)))
        self.block_ids_per_group[group_id] = block_id

    def remove_block_id(self, block_id: int) -> bool:
        """Unregister a physical block (by value, across all groups).

        Returns True if the block was present and cleared; False if it
        was not found (no-op — happens on duplicate free or when the hash
        was never materialized in any block).
        """
        found = False
        for gid, bid in enumerate(self.block_ids_per_group):
            if bid == block_id:
                self.block_ids_per_group[gid] = None
                found = True
        return found

    def snapshot_block_ids(self) -> list[int | None]:
        """Copy of ``block_ids_per_group`` (SSM classify 时快照用)."""
        return list(self.block_ids_per_group)

    def clear_block_ids(self) -> None:
        """Clear all physical block bindings, keeping per-group shape."""
        self.block_ids_per_group = [None] * len(self.block_ids_per_group)


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
        apply_evict_remote_cache_fn: Callable[[str, list[BlockHash]], int | None] | None = None,
        apply_evict_kvblock_fn: Callable[[str, list[tuple[BlockHashState, list[int | None]]], bool], list[BlockHash]] | None = None,
    ):
        logger.info(
            "SSM: __init__ max_sessions=%d shared_session_hash_state=%s shared_hash_state=%s "
            "has_evict_remote_cb=%s has_evict_kvblock_cb=%s",
            max_sessions,
            session_hash_state is not None,
            hash_state is not None,
            apply_evict_remote_cache_fn is not None,
            apply_evict_kvblock_fn is not None,
        )
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
        logger.debug(
            "SSM: __init__ done active_session_ids=%s lru_len=%d total_hash_state=%d",
            self._active_session_ids,
            len(self._lru),
            len(self._hash_state),
        )

    # ════════════════════════════════════════════════════════════════
    # Active session tracking
    # ════════════════════════════════════════════════════════════════

    def mark_session_active(self, session_id: str) -> None:
        logger.debug("SSM: mark_session_active enter [%s] current_active=%s", session_id, self._active_session_ids)
        if session_id not in self._active_session_ids:
            logger.info("SSM: mark_session_active [%s]", session_id)
            self._active_session_ids.add(session_id)
        else:
            logger.debug("SSM: mark_session_active [%s] already active, no-op", session_id)

    def mark_session_inactive(self, session_id: str) -> None:
        logger.debug("SSM: mark_session_inactive enter [%s] current_active=%s", session_id, self._active_session_ids)
        if session_id in self._active_session_ids:
            logger.info("SSM: mark_session_inactive [%s]", session_id)
            self._active_session_ids.discard(session_id)
        else:
            logger.debug("SSM: mark_session_inactive [%s] not active, no-op", session_id)

    def _is_session_active(self, session_id: str) -> bool:
        is_active = session_id in self._active_session_ids
        logger.debug("SSM: _is_session_active [%s] -> %s", session_id, is_active)
        return is_active

    def set_session_block_hashes(
        self, session_id: str, new_hashes: list[BlockHash],
    ) -> None:
        """Diff-register hashes: update BlockHashState ref_count/ref_sessions
        and rewrite the shared ``_session_hash_state[session_id]`` list.

        Diff strategy (only handle added/removed, leave common alone):
          - removed (old - new): decrement ref_count; if 0, del _hash_state.
          - added   (new - old): create state or add_ref (resets offload_count).
          - common  (old & new): untouched — avoids resetting offload_count
            of a long-held hash or triggering state delete-then-recreate.

        调用时点：scheduler 在 on_request_scheduled 中对本轮请求的全部
        hash 调用（先于 allocate/cache-hit 回调），因此
        ``BlockHashState.block_ids_per_group`` 此时尚未填充（全 None），
        由 SAM 在后续回调中填入实际 block_id。
        """
        old_states = self._session_hash_state.get(session_id, [])
        old_hashes = {s.block_hash for s in old_states}
        new_set = set(new_hashes)

        removed = [s for s in old_states if s.block_hash not in new_set]
        added = [h for h in new_hashes if h not in old_hashes]
        logger.debug(
            "SSM: set_session_block_hashes enter [%s] old=%d new=%d added=%d removed=%d common=%d",
            session_id, len(old_states), len(new_hashes), len(added), len(removed),
            len(old_hashes & new_set),
        )
        if removed or added:
            logger.debug("SSM: set_session_block_hashes [%s] %d -> %d hashes (added=%d, removed=%d)",
                         session_id, len(old_states), len(new_hashes), len(added), len(removed))
        else:
            logger.debug("SSM: set_session_block_hashes [%s] no diff, common=%d", session_id, len(old_hashes))

        for state in removed:
            # removed：仅 state 层减 ref / 归零删 entry（SSM 是 state 权威）。
            # block 侧旧引用不在此处理 —— 由 block 复用清理路径
            # （SAM.clear_block_session_refs）兜底收敛。
            if state.remove_ref(session_id):
                del self._hash_state[state.block_hash]

        new_count = 0
        reset_count = 0
        new_state_list: list[BlockHashState] = []
        for h in new_hashes:
            state = self._hash_state.get(h)
            if h in old_hashes:
                # common：沿用旧 state 对象（同一份引用，block_ids 保留）
                assert state is not None
            elif state is not None:
                # added 且全局已存在（其他 session 引用过）：add_ref
                if state.add_ref(session_id):
                    reset_count += 1
            else:
                state = BlockHashState(
                    block_hash=h, ref_count=1, ref_sessions={session_id},
                    offload_count=0,
                )
                self._hash_state[h] = state
                new_count += 1
            new_state_list.append(state)

        # SAM 侧写 _session_hashes：SSM 是共享表的权威维护者（设计 §3.1），
        # list 保序 = hash_block_size 粒度的位置顺序。
        self._session_hash_state[session_id] = new_state_list

        if new_count or reset_count:
            logger.debug("SSM: set_session_block_hashes [%s] %d new global entries, %d reset to HBM, total managed hashes=%d",
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
        is advanced via ``offload_once()`. ``block_ids_per_group`` 仅做快照
        不清空 —— offload 不动 block.sessions，HBM 物理绑定仍然有效。

        Returns OffloadWork dict:
          - ``hbm_work``:   [(state, block_ids_per_group_snapshot) for state in first_offload_states]
                            offload_count 0→1, Controller → sam.apply_offload_kvblock
          - ``remote_work``: [state.block_hash for state in second_offload_states]
                            offload_count 1→2, Controller → spm.on_context_management_offload
          - ``skipped``:    [state for state in active_shared_or_max_reached]
                            no change
        """
        logger.debug(
            "SSM: classify_offload_work enter [%s] states=%d active_sessions=%s",
            session_id, len(states), self._active_session_ids,
        )
        hbm_work: list[tuple[BlockHashState, list[int | None]]] = []
        remote_work: list[BlockHash] = []
        skipped: list[BlockHashState] = []
        skip_reasons: dict[str, int] = {}
        for state in states:
            if state.is_shared_with_active_session(session_id, self._is_session_active):
                skipped.append(state)
                skip_reasons["active_share"] = skip_reasons.get("active_share", 0) + 1
                continue
            if state.offload_count >= state.max_offload_count:
                skipped.append(state)
                skip_reasons["max_reached"] = skip_reasons.get("max_reached", 0) + 1
                continue
            reached_bottom = state.offload_once()
            # 仅快照、不清空 block_ids_per_group：offload 不动 block.sessions
            # （session 仍逻辑持有 block，HBM 物理绑定仍然有效），后续 evict
            # 仍需通过绑定定位 block 清理引用。apply_offload_kvblock 只消费
            # hash（TTL 保护注册），不依赖 block 快照。
            block_ids_snapshot = state.snapshot_block_ids()
            if reached_bottom:
                remote_work.append(state.block_hash)
            else:
                hbm_work.append((state, block_ids_snapshot))

        logger.info("SSM: classify_offload_work [%s] hbm=%d remote=%d skipped=%d",
                     session_id, len(hbm_work), len(remote_work), len(skipped))
        if skip_reasons:
            logger.debug("SSM: classify_offload_work [%s] skip_breakdown=%s", session_id, skip_reasons)
        self._lru.touch(session_id)
        return {"hbm_work": hbm_work, "remote_work": remote_work, "skipped": skipped}

    # ════════════════════════════════════════════════════════════════
    # Classify evict work — Controller 调用
    # ════════════════════════════════════════════════════════════════

    def classify_evict_work(
        self,
        session_id: str,
        states: list[BlockHashState],
        is_session_op: bool | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, list[tuple[BlockHashState, list[int | None]]]]:
        logger.debug(
            "SSM: classify_evict_work enter [%s] states=%d is_session_op=%s range=[%s,%s) "
            "session_hash_len=%d active_sessions=%s",
            session_id, len(states), is_session_op, start, end,
            len(self._session_hash_state.get(session_id, [])),
            self._active_session_ids,
        )
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

        ``is_session_op`` / ``start`` / ``end``：由调用方（Controller）传入
        SAM 切片时返回的扩展区间，SSM 据此 trim ``_session_hash_state[sid]``
        （range evict 按 [start, end) 切片剔除）或 pop 整条（session evict，
        同时清 LRU 条目）。未显式传入时退回按覆盖长度推断（LRU 驱逐传整条
        session states，等价 session op）。
        """
        hbm_only_work: list[tuple[BlockHashState, list[int | None]]] = []
        hbm_and_remote_work: list[tuple[BlockHashState, list[int | None]]] = []
        skipped: list[BlockHashState] = []
        for state in states:
            if state.is_shared_with_active_session(session_id, self._is_session_active):
                skipped.append(state)
                continue
            state.ref_sessions.discard(session_id)
            state.ref_count = max(0, state.ref_count - 1)
            block_ids_snapshot = state.snapshot_block_ids()
            if state.ref_count == 0:
                # 独占 hash：state 即将删除，绑定一并清空
                state.clear_block_ids()
                del self._hash_state[state.block_hash]
                hbm_and_remote_work.append((state, block_ids_snapshot))
            else:
                # 共享 hash：保留 block_ids_per_group 绑定 —— block 仍被
                # 其他 session 持有，最后一个 session 驱逐时（ref 归 0）
                # SAM 仍需通过绑定定位 block 清理引用
                hbm_only_work.append((state, block_ids_snapshot))

        # SSM bookkeeping on _session_hash_state + LRU.
        # skipped（active-share 保护）的 state 保留在 session 列表中：
        # 其 ref_count/ref_sessions 未变（sid 仍引用该 hash），若从列表
        # 摘除会导致该引用永不释放（ref 永不归 0 的僵尸条目）。
        skipped_ids = {id(s) for s in skipped}
        if is_session_op is None:
            session_states = self._session_hash_state.get(session_id, [])
            is_session_op = len(states) >= len(session_states)
        if is_session_op:
            if skipped_ids:
                # 部分 state 被 active-share 保护：仅移除非 skipped 的，
                # session 仍持有这些 hash（后续可重试 evict）
                remaining_states = [s for s in states if id(s) in skipped_ids]
                self._session_hash_state[session_id] = remaining_states
                self._lru.touch(session_id)
                self._lru.set_hash_count(session_id, len(remaining_states))
            else:
                # session evict: pop 整条 + 清 LRU 条目
                self._session_hash_state.pop(session_id, None)
                self._lru.remove(session_id)
        elif start is not None and end is not None:
            # range evict: 按 SAM 返回的扩展区间 [start, end) 切片剔除
            # （排除 skipped），保证 trim 范围与物理 block 边界对齐
            old = self._session_hash_state.get(session_id)
            if old is not None:
                self._session_hash_state[session_id] = (
                    old[:start]
                    + [s for s in old[start:end] if id(s) in skipped_ids]
                    + old[end:]
                )
            remaining = self._session_hash_state.get(session_id)
            if remaining:
                self._lru.touch(session_id)
                self._lru.set_hash_count(session_id, len(remaining))
        else:
            # 防御路径（未传区间）：用 block_hash 集合做差集（排除
            # skipped），避免残留已被 del _hash_state 的僵尸引用。
            evicted_hashes = {s.block_hash for s in states if id(s) not in skipped_ids}
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
        logger.debug(
            "SSM: process_prefetch_hashes enter [%s] block_hashes=%d hash_state_size=%d lru_managed=%s",
            session_id, len(block_hashes), len(self._hash_state),
            self._lru.is_managed(session_id),
        )
        if not block_hashes:
            logger.debug("SSM: process_prefetch_hashes [%s] empty input, no-op", session_id)
            return
        reset_count = 0
        not_found_count = 0
        for h in block_hashes:
            state = self._hash_state.get(h)
            if state is None:
                not_found_count += 1
                continue
            if state.offload_count != 0:
                state.offload_count = 0
                reset_count += 1
        logger.info("SSM: process_prefetch_hashes [%s] reset %d / %d hashes to HBM",
                     session_id, reset_count, len(block_hashes))
        if not_found_count:
            logger.debug("SSM: process_prefetch_hashes [%s] %d hashes not found in _hash_state",
                         session_id, not_found_count)
        if self._lru.is_managed(session_id):
            self._lru.touch(session_id)

    def _try_evict_if_needed(self) -> None:
        """Proactively evict inactive sessions until the LRU is no longer full.

        Drives one session per iteration; re-checks ``is_full`` after each
        eviction so a multi-session overflow is fully drained in one call.
        Stops early if no inactive candidate is available (avoids spinning
        while all remaining sessions are active).
        """
        if not self._lru.is_full():
            logger.debug("SSM: _try_evict_if_needed LRU not full (lru=%d max=%d), no-op",
                         len(self._lru), self._lru.max_sessions)
            return
        logger.info(
            "SSM: _try_evict_if_needed enter lru_len=%d max=%d is_full=%s active_count=%d",
            len(self._lru), self._lru.max_sessions, self._lru.is_full(),
            len(self._active_session_ids),
        )
        remaining_budget = len(self._lru)
        while self._lru.is_full() and remaining_budget > 0:
            remaining_budget -= 1
            candidates = self._lru.get_evict_candidates(
                count=1, active_session_ids=self._active_session_ids)
            if not candidates:
                logger.debug(
                    "SSM: _try_evict_if_needed no inactive candidate (lru=%d active=%d), stop",
                    len(self._lru), len(self._active_session_ids),
                )
                return

            sid = candidates[0]
            logger.info("SSM: _try_evict_if_needed LRU full (%d >= %d), evicting [%s]",
                         len(self._lru), self._lru.max_sessions, sid)

            # 1. slice states (whole session)
            states = self._session_hash_state.get(sid, [])
            if not states:
                # 空 session（短请求未满一个 hash block）：pop 残留的空列表
                # 键 + 清 LRU 条目
                self._session_hash_state.pop(sid, None)
                self._lru.remove(sid)
                continue

            # 2. classify + mutate state (pop _session_hash_state + remove LRU inside)
            #    LRU 驱逐是 session-evict 的特例（SSM 触发，无 Controller 介入），
            #    显式传 is_session_op=True
            evict_work = self.classify_evict_work(sid, states, is_session_op=True)
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
                self._apply_evict_remote_cache_fn(sid, freed_hashes)

            logger.info(
                "SSM: _try_evict_if_needed [%s] done — states=%d hbm_only=%d "
                "hbm_and_remote=%d skipped=%d freed_hashes=%d (spm_notified=%s)",
                sid, len(states), len(hbm_only_work), len(hbm_and_remote_work),
                len(evict_work.get("skipped", [])), len(freed_hashes),
                bool(freed_hashes) and self._apply_evict_remote_cache_fn is not None,
            )

