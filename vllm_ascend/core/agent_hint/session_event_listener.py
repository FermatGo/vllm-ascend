from typing import Protocol

from vllm.v1.core.kv_cache_utils import BlockHash


class SessionEventListener(Protocol):
    """SPM 实现此协议，监听 SAM 的 session 生命周期事件"""

    def on_session_registered(self, session_id: str, parent_session_id: str | None) -> None: ...

    def on_session_blocks_protected(
        self,
        block_hashes: list[BlockHash],
    ) -> None: ...

    def on_session_blocks_removed(
        self,
        block_hashes: list[BlockHash],
    ) -> int: ...

    # def on_session_cache_hit(
    #     self,
    #     block_hashes: list[BlockHash],
    # ) -> None: ...

    # def on_session_ttl_expired(self, block_hashs: list[BlockHash]) -> None: ...

    def on_session_freed(self, session_id: str) -> None: ...

    # def on_context_management_evict(
    #     self,
    #     block_hashes: list[BlockHash],
    # ) -> None: ...

    def on_context_management_offload(self, session_id: str, block_ids: list[int]) -> None: ...

    # def on_context_management_prefetch(
    #     self,
    #     session_id: str,
    #     block_hashes: list[BlockHash],
    # ) -> bool: ...

    def on_check_matched_token(
        self,
        session_id: str,
        check_matched_start: int,
        check_matched_end: int,
    ) -> int:
        ...

        #############################################################################

    def on_session_blocks_allocated(
        self,
        session_id: str | None,
        block_ids: list[int],
        pool_keys: list[str],  # 远端 PoolKey 列表（来自 AscendStoreConnector）
        block_hashes: list[BlockHash],  # 对应的 block hash
    ) -> None: ...

    def on_session_cache_hit(
        self,
        session_id: str | None = None,
        block_id: int | None = None,
        # pool_key: str | None,      # cache hit 的 block 对应的远端 PoolKey（可能无）
        block_hash: BlockHash | None = None,
    ) -> None: ...

    def on_context_management_prefetch(
        self,
        session_id: str | None = None,
        logical_block_start: int | None = None,
        logical_block_end: int | None = None,
        block_ids: list[int] | None = None,
        block_hashes: list[BlockHash] = None,
    ) -> bool: ...

    def on_context_management_evict(
        self,
        session_id: str | None = None,
        block_ids: list[int] = None,
        block_hashes: list[BlockHash] = None,
    ) -> int: ...

    def on_session_ttl_expired(
        self,
        session_id: str | None = None,
        block_ids: list[int] = None,
        block_hash: BlockHash | None = None,
    ) -> int: ...
