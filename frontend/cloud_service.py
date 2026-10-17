
from functools import lru_cache

from app.agents.workflow import create_agent_workflow
from app.llm.provider import get_llm_client
from app.memory.conversation_service import ConversationService
from app.memory.conversation_store import ConversationStore
from app.rag.rag_pipeline import RAGPipeline


@lru_cache(maxsize=1)
def get_cloud_service():
    llm_client = get_llm_client()

    rag_pipeline = RAGPipeline(
        llm_client=llm_client
    )

    agent = create_agent_workflow(
        rag_pipeline=rag_pipeline,
        llm_client=llm_client,
    )

    return ConversationService(
        store=ConversationStore(
            max_messages=10,
            max_sessions=100,
        ),
        agent=agent,
        llm_client=llm_client,
    )
