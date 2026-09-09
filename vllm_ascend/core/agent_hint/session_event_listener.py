from typing import Protocol
from vllm.v1.core.kv_cache_utils import BlockHash


class SessionEventListener(Protocol):
    """SPM 实现此协议，监听 SAM 的 session 生命周期事件"""

    def on_session_blocks_protected(
        self,
        session_id,
        block_hashes: list[BlockHash],
    ) -> None: ...

    def on_session_blocks_removed(
        self,
        session_id,
        block_hashes: list[BlockHash],
    ) -> int: ...

    def on_context_management_offload(
        self,
        block_hashes: list[BlockHash] = None,
    ) -> None: ...

    def on_session_blocks_allocated(
        self,
        session_id: str | None,
        block_hashes: list[BlockHash],    # 对应的 block hash
    ) -> None: ...

    def on_session_cache_hit(
        self,
        session_id: str | None = None,
        block_id: int | None = None,
        block_hash: BlockHash | None = None,
    ) -> None: ...

    def on_context_management_prefetch(
        self,
        session_id: str | None = None,
        block_hashes: list[BlockHash] = None,
    ) -> bool: ...

    def on_context_management_evict(
        self,
        block_hashes: list[BlockHash] = None,
    ) -> int: ...

    def on_session_ttl_expired(
        self,
        session_id: str | None = None,
        block_hashes: list[BlockHash] | None = None,
    ) -> int: ...