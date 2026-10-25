import logging
from pathlib import Path
from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import (
    wrap_tool_call, before_model, dynamic_prompt, ModelRequest,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from config.settings import settings


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _load_prompt(path_str:str)->str:
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.read_text(encoding="utf-8")

@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    logger.info("[tool] 执行: %s 参数: %s", request.tool_call["name"], request.tool_call["args"])
    try:
        result = handler(request)
        logger.info("[tool] %s 调用成功", request.tool_call["name"])
        if request.tool_call["name"] == "fill_context_for_report":
            request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.error("[tool] %s 失败: %s", request.tool_call["name"], e)
        raise

@before_model
def log_before_model(
        state:AgentState,
        runtime:Runtime,
):
    logger.info("[model] 即将调用，消息数=%d", len(state["messages"]))
    logger.debug("[model] 末条: %s | %s", type(state["messages"][-1]).__name__, state["messages"][-1].content)
    return None

@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    return _load_prompt(settings.REPORT_PROMPT_PATH) if is_report else _load_prompt(settings.SYSTEM_PROMPT_PATH)



