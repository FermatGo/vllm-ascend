#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Ascend implementation of vLLM agent-hint cache management."""
 
from vllm_ascend.core.agent_hint.backend import (
    ASCEND_AGENT_HINT_BACKEND,
    AscendAgentHintBackend,
)
from vllm_ascend.core.agent_hint.free_kv_cache_block_queue import (
    AgentHintFreeKVCacheBlockQueue,
)
from vllm_ascend.core.agent_hint.manager import AscendAgentHintManager
 
__all__ = [
    "ASCEND_AGENT_HINT_BACKEND",
    "AgentHintFreeKVCacheBlockQueue",
    "AscendAgentHintBackend",
    "AscendAgentHintManager",
]