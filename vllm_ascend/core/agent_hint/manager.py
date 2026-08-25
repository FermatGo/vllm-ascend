#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Scheduler-side composition of Ascend session-aware cache managers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.core.agent_hint_manager import (
    AgentHintManager,
    AgentHintManagerContext,
)
from vllm.v1.engine import AgentHintResponse

from vllm_ascend.core.agent_hint.session_aware_manager import (
    SessionAwareManager,
)
from vllm_ascend.core.agent_hint.session_aware_pooling_manager import (
    SessionAwarePoolingManager,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request


class AscendAgentHintManager(AgentHintManager):
    """Owns the Ascend session, TTL, offload and prefetch policies."""

    def __init__(self, context: AgentHintManagerContext) -> None:
        self._session_manager = SessionAwareManager(context.kv_cache_manager)
        self._pooling_manager: SessionAwarePoolingManager | None = None

        if context.connector is not None:
            self._pooling_manager = SessionAwarePoolingManager(
                sam=self._session_manager,
                add_request=context.add_request,
                connector=context.connector,
            )
            self._pooling_manager.block_size = context.hash_block_size
            self._pooling_manager.start()

        context.kv_cache_manager.register_session_event_callbacks(
            on_blocks_allocated=(self._session_manager.on_blocks_allocated_for_request),
            on_block_cache_hit=(self._session_manager.on_block_cache_hit_for_request),
        )

    def is_kvc_management_request(self, request: Request) -> bool:
        hint = request.agent_hint
        return bool(hint and hint.context_management and hint.context_management.manage_request)

    def register_kvc_management_request(self, request: Request) -> AgentHintResponse | None:
        hint = request.agent_hint
        if hint is None or hint.context_management is None:
            return None
        edit_results = self._session_manager.register_agent_hint(
            request.request_id,
            hint.session_id,
            hint.context_management,
        )
        return AgentHintResponse(
            session_id=hint.session_id,
            edit_results=edit_results,
        )

    def on_request_added(self, request: Request) -> None:
        hint = request.agent_hint
        self._session_manager.register_agent_hint(
            request.request_id,
            hint.session_id if hint else None,
            hint.context_management if hint else None,
        )

    def on_request_scheduled(self, request: Request) -> None:
        hint = request.agent_hint
        if hint is not None and hint.session_id:
            self._session_manager.register_session_block_hash(hint.session_id, request.block_hashes)

    def on_step(self, num_unfinished_requests: int) -> None:
        self._session_manager.tick()
        if self._pooling_manager is not None:
            self._pooling_manager.process_prefetch_req(num_unfinished_requests=num_unfinished_requests)

    def on_request_finished(self, request: Request) -> None:
        self._session_manager.on_request_completed(request.request_id)

    def has_pending_work(self) -> bool:
        return bool(self._pooling_manager and self._pooling_manager.prefetch_waiting_queue)

    def shutdown(self) -> None:
        if self._pooling_manager is not None:
            self._pooling_manager.stop()
