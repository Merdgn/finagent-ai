
from functools import lru_cache

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field

from app.rag.rag_pipeline import RAGPipeline
from app.agents.workflow import create_agent_workflow


app = FastAPI(
    title="FinAgent AI",
    description="Secure Banking Conversational AI Assistant",
    version="0.3.0",
)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieved_chunks: int


class AgentResponse(BaseModel):
    answer: str
    route: str
    sources: list[str]


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline()


@lru_cache(maxsize=1)
def get_agent_workflow():
    return create_agent_workflow(get_rag_pipeline())


@app.get("/")
def home():
    return {
        "project": "FinAgent AI",
        "message": "Welcome to FinAgent AI",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    try:
        result = pipeline.ask(request.question)

        return ChatResponse(
            answer=result.answer,
            sources=result.sources,
            retrieved_chunks=result.retrieved_chunks,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable.",
        ) from exc


@app.post("/agent/chat", response_model=AgentResponse)
def agent_chat(
    request: ChatRequest,
    agent=Depends(get_agent_workflow),
):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        result = agent.invoke({
            "question": question
        })

        return AgentResponse(
            answer=result["answer"],
            route=result["route"],
            sources=result.get("sources", []),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="Agent service temporarily unavailable.",
        ) from exc
