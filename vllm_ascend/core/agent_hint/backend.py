# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified Ascend Agent Hint backend."""
 
from vllm.v1.core.agent_hint_manager import (
    AgentHintBackend,
    AgentHintManager,
    AgentHintManagerContext,
)
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
 
from vllm_ascend.core.agent_hint.free_kv_cache_block_queue import (
    AgentHintFreeKVCacheBlockQueue,
)
from vllm_ascend.core.agent_hint.manager import AscendAgentHintManager
 
 
class AscendAgentHintBackend(AgentHintBackend):
    """Provide the Ascend Manager and three-zone free-block queue."""
 
    def is_supported(self) -> bool:
        from vllm.platforms import current_platform
 
        return type(current_platform).__module__.startswith("vllm_ascend.")
 
    def create_manager(self, context: AgentHintManagerContext) -> AgentHintManager:
        connector = context.connector
        if connector is not None and not type(connector).__module__.startswith(
            "vllm_ascend."
        ):
            return AgentHintManager()
        return AscendAgentHintManager(context)
 
    def create_free_kv_cache_block_queue(
        self,
        blocks: list[KVCacheBlock],
    ) -> FreeKVCacheBlockQueue:
        return AgentHintFreeKVCacheBlockQueue(blocks)
 
 
ASCEND_AGENT_HINT_BACKEND = AscendAgentHintBackend()