from dataclasses import dataclass
from typing import Any, Literal
from collections.abc import Mapping


@dataclass
class ContextManagementEditsParams:  # 上下文编辑
    type: Literal["offload", "prefetch", "evict"] = "offload"
    start: int | None = None  # 起始 message index（pymotor 侧语义）
    end: int | None = None  # 结束 message index（pymotor 侧语义）
    target: Literal["session", "messages", "tools"] = "messages"
    block_start: int | None = None  # pymotor 转换的起始 block index
    block_end: int | None = None  # pymotor 转换的结束 block index


@dataclass
class ContextManagementParams:
    manage_request: bool | None = False  # 仅kvc管理请求，出现该字段表示请求本身内容并不会被执行
    edits: list[ContextManagementEditsParams] | None = None


@dataclass
class CacheControlParams:
    type: Literal["ephemeral"] = "ephemeral"  # 仅支持 ephemeral
    ttl: float = 300.0  # 缓存保留时间（秒），默认5min，最大1h
    msg_offset: int | None = None
    block_offset: int | None = None
    token_offset: int | None = None


@dataclass
class AgentHintParams:
    """Ascend 侧解析后的 agent_hint 强类型表示。

    HTTP/引擎层 ``agent_hint`` 现在统一为 ``dict`` 透传；本类仅作为
    ascend 内部使用的强类型视图，由 ``convert_agent_hint_dict`` 从 dict
    构造。
    """
    session_id: str | None = None
    parent_session_id: str | None = None
    cache_control: CacheControlParams | None = None
    context_management: ContextManagementParams | None = None
    latency_control: dict | None = None
    priority_control: dict | None = None


def _to_mapping(obj: Any) -> Mapping:
    """Coerce a pydantic model / mapping / arbitrary object into a mapping."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, Mapping):
        return obj
    return dict(obj)


def convert_agent_hint_dict(agent_hint: Any) -> AgentHintParams | None:
    """把 ``EngineCoreRequest`` / ``Request`` 透传的 opaque ``agent_hint``
    (dict 或 pydantic 模型) 解析为 ascend 内部使用的强类型
    ``AgentHintParams``。

    ``agent_hint`` 为 ``None`` 时返回 ``None``。
    """
    if agent_hint is None:
        return None

    # 已经是 ascend 侧的强类型 dataclass，直接返回。
    if isinstance(agent_hint, AgentHintParams):
        return agent_hint

    ah = _to_mapping(agent_hint)

    cache_control = ah.get("cache_control")
    if cache_control is not None and not isinstance(
        cache_control, CacheControlParams
    ):
        cc = _to_mapping(cache_control)
        cache_control = CacheControlParams(
            type=cc.get("type", "ephemeral"),
            ttl=cc.get("ttl", 300.0),
            msg_offset=cc.get("msg_offset"),
            block_offset=cc.get("block_offset"),
            token_offset=cc.get("token_offset"),
        )

    context_management = ah.get("context_management")
    if context_management is not None and not isinstance(
        context_management, ContextManagementParams
    ):
        cm = _to_mapping(context_management)
        raw_edits = cm.get("edits") or []
        edits: list[ContextManagementEditsParams] = []
        for raw_edit in raw_edits:
            if isinstance(raw_edit, ContextManagementEditsParams):
                edits.append(raw_edit)
                continue
            e = _to_mapping(raw_edit)
            edits.append(
                ContextManagementEditsParams(
                    type=e.get("type", "offload"),
                    start=e.get("start", 0),
                    end=e.get("end", 0),
                    target=e.get("target", "messages"),
                    block_start=e.get("block_start"),
                    block_end=e.get("block_end"),
                )
            )
        context_management = ContextManagementParams(
            manage_request=cm.get("manage_request", False),
            edits=edits,
        )

    return AgentHintParams(
        session_id=ah.get("session_id"),
        parent_session_id=ah.get("parent_session_id"),
        cache_control=cache_control,
        context_management=context_management,
        latency_control=ah.get("latency_control"),
        priority_control=ah.get("priority_control"),
    )
