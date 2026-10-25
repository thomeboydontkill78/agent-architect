from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from fastapi.responses import StreamingResponse
from core.agent import get_agent, DEFAULT_CONTEXT


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 初始化 Agent...")
    app.state.agent = get_agent()
    print("✅ Agent 就绪")
    yield


app = FastAPI(title="Agent Architect API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskRequest, request: Request):
    agent = request.app.state.agent
    result = agent.invoke(
        {"messages": [("user", req.question)]},
        context=DEFAULT_CONTEXT,
    )
    return {"answer": result["messages"][-1].content}

@app.post("/stream")
def stream(req: AskRequest, request: Request):
    agent = request.app.state.agent

    def gen():
        for chunk in agent.stream(
            {"messages": [("user", req.question)]},
            context=DEFAULT_CONTEXT,
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                if not update or "messages" not in update:
                    continue
                for msg in update["messages"]:
                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls:
                        names = ", ".join(tc["name"] for tc in tool_calls)
                        yield f"data: {json.dumps({'type':'tool','text':names}, ensure_ascii=False)}\n\n"
                    elif msg.content and node == "model":
                        yield f"data: {json.dumps({'type':'answer','text':msg.content}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")