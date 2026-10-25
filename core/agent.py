from langchain.agents import create_agent
from core.model import get_chat_model
from tools.agent_tools import (
    rag_summarize, get_weather, get_user_location,
    get_user_id, get_current_month, fetch_external_data,
    fill_context_for_report,
)
from middleware.agent_middleware import (
    monitor_tool, log_before_model, report_prompt_switch,
)

TOOLS = [
    rag_summarize, get_weather, get_user_location,
    get_user_id, get_current_month, fetch_external_data,
    fill_context_for_report,
]
MIDDLEWARE = [monitor_tool, log_before_model, report_prompt_switch]


DEFAULT_CONTEXT = {"report": False}


def get_agent():
    return create_agent(
        model=get_chat_model(),
        tools=TOOLS,
        middleware=MIDDLEWARE,
        context_schema=dict,
    )

