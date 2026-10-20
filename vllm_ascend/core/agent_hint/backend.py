# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified Ascend Agent Hint backend."""

from vllm.v1.core.agent_hint_manager import (
    AgentHintBackend,
    AgentHintManager,
    AgentHintManagerContext,
)
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

from vllm_ascend import envs
from vllm_ascend.core.agent_hint.free_kv_cache_block_queue import (
    AgentHintFreeKVCacheBlockQueue,
    AgentHintKVCacheBlock,
)
from vllm_ascend.core.agent_hint.manager import AscendAgentHintManager


class AscendAgentHintBackend(AgentHintBackend):
    """Provide the Ascend Manager and three-zone free-block queue."""

    def is_supported(self) -> bool:
        if not envs.VLLM_ASCEND_ENABLE_AGENT_HINT:
            return False

        from vllm.platforms import current_platform

        return type(current_platform).__module__.startswith("vllm_ascend.")

    def create_agent_hint_manager(self, context: AgentHintManagerContext) -> AgentHintManager:
        if not context.vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Agent Hint requires prefix caching; enable it with --enable-prefix-caching.")

        connector = context.connector
        if connector is not None:
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
                AscendStoreConnector,
            )

            if not isinstance(connector, AscendStoreConnector):
                raise ValueError(
                    "Agent Hint remote cache management requires AscendStoreConnector; "
                    f"got {type(connector).__module__}.{type(connector).__name__}."
                )
        return AscendAgentHintManager(context)

    def create_kv_cache_block(self, block_id: int) -> KVCacheBlock:
        return AgentHintKVCacheBlock(block_id)

    def create_free_kv_cache_block_queue(
        self,
        blocks: list[KVCacheBlock],
    ) -> FreeKVCacheBlockQueue:
        return AgentHintFreeKVCacheBlockQueue(blocks)


ASCEND_AGENT_HINT_BACKEND = AscendAgentHintBackend()
