from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
