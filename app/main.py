
from functools import lru_cache

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field


from uuid import UUID

from app.memory.conversation_store import ConversationStore
from app.memory.conversation_service import ConversationService

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


class ConversationRequest(BaseModel):
    session_id: str
    question: str = Field(
        min_length=1,
        max_length=2000,
    )


class ConversationAPIResponse(BaseModel):
    session_id: str
    answer: str
    route: str
    sources: list[str]
    standalone_question: str


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline()


@lru_cache(maxsize=1)
def get_agent_workflow():
    return create_agent_workflow(get_rag_pipeline())


@lru_cache(maxsize=1)
def get_conversation_store() -> ConversationStore:
    return ConversationStore(
        max_messages=10,
        max_sessions=100,
    )


@lru_cache(maxsize=1)
def get_conversation_service() -> ConversationService:
    return ConversationService(
        store=get_conversation_store(),
        agent=get_agent_workflow(),
        llm_client=get_rag_pipeline().llm_client,
    )


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



@app.post(
    "/agent/conversation",
    response_model=ConversationAPIResponse,
)
def agent_conversation(
    request: ConversationRequest,
    service: ConversationService = Depends(
        get_conversation_service
    ),
):
    try:
        session_id = str(UUID(request.session_id))

        result = service.ask(
            session_id=session_id,
            question=request.question,
        )

        return ConversationAPIResponse(
            session_id=result.session_id,
            answer=result.answer,
            route=result.route,
            sources=result.sources,
            standalone_question=result.standalone_question,
        )

    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid session ID or question.",
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="Conversation service unavailable.",
        ) from exc
