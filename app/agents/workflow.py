
import re
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict, total=False):
    question: str
    route: str
    answer: str
    sources: list[str]


BANKING_TERMS = (
    "bank", "account", "card", "credit",
    "savings", "interest", "loan", "payment",
    "pin", "withdrawal", "deposit",
)


CONVERSATION_TERMS = {
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "how are you",
    "merhaba",
    "selam",
    "nasılsın",
    "nasılsın?",
    "teşekkürler",
    "teşekkür ederim",
    "günaydın",
    "iyi akşamlar",
}

CHAT_SYSTEM_PROMPT = """
You are FinAgent AI, a friendly and professional
banking information assistant.

Respond naturally in the user's language.

Rules:
- Be conversational, concise and helpful.
- Do not invent banking rates, fees or policies.
- Never request passwords, PINs or verification codes.
- Do not provide personalized financial advice.
- Direct product-specific banking questions to
  official documents instead of guessing.
- Do not claim affiliation with a real bank.
"""


def create_agent_workflow(rag_pipeline, llm_client=None):
    
    
    def router_node(state: AgentState) -> dict:
        question = state["question"].strip().lower()

        # Preserve explicit calculator commands.
        if question.startswith("calculate:"):
            return {"route": "calculator"}

        # Banking questions take priority.
        if any(
            re.search(rf"\b{re.escape(term)}\b", question)
            for term in BANKING_TERMS
        ):
            return {"route": "rag"}

        # Recognize common conversational expressions.
        normalized = re.sub(
            r"[!?.,;:]+", " ", question
        )
        normalized = " ".join(normalized.split())

        greetings = (
            "merhaba",
            "selam",
            "nasılsın",
            "günaydın",
            "iyi akşamlar",
            "hello",
            "hi",
            "hey",
            "how are you",
            "teşekkür",
            "thanks",
            "thank you",
        )

        if llm_client is not None and any(
            normalized == phrase
            or normalized.startswith(phrase + " ")
            or normalized.endswith(" " + phrase)
            for phrase in greetings
        ):
            return {"route": "conversation"}

        return {"route": "safety"}



    def select_route(state: AgentState) -> str:
        return state["route"]

    def rag_node(state: AgentState) -> dict:
        result = rag_pipeline.ask(state["question"])

        return {
            "answer": result.answer,
            "sources": result.sources,
        }

    def calculator_node(state: AgentState) -> dict:
        expression = state["question"].split(":", 1)[1].strip()

        # Only simple arithmetic; never execute user code.
        pattern = (
            r"^(-?\d+(?:\.\d+)?)\s*"
            r"([+\-*/])\s*"
            r"(-?\d+(?:\.\d+)?)$"
        )

        match = re.fullmatch(pattern, expression)

        if not match:
            return {
                "answer": (
                    "Use: calculate: 100 + 20 "
                    "(+, -, *, / supported)."
                ),
                "sources": [],
            }

        left_text, operator, right_text = match.groups()

        try:
            left = Decimal(left_text)
            right = Decimal(right_text)

            if operator == "+":
                result = left + right
            elif operator == "-":
                result = left - right
            elif operator == "*":
                result = left * right
            else:
                result = left / right

            answer = f"Result: {result}"

        except (DivisionByZero, InvalidOperation):
            answer = "Invalid arithmetic operation."

        return {"answer": answer, "sources": []}


    def conversation_node(state: AgentState) -> dict:
        if llm_client is None:
            return {
                "answer": "Conversation model unavailable.",
                "sources": [],
            }

        answer = llm_client.generate(
            prompt=state["question"],
            system_prompt=CHAT_SYSTEM_PROMPT,
        )

        return {
            "answer": answer,
            "sources": [],
        }


    def safety_node(state: AgentState) -> dict:
        return {
            "answer": (
                "FinAgent AI currently supports banking "
                "document questions and basic calculations. "
                "Do not share personal banking credentials."
            ),
            "sources": [],
        }

    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("safety", safety_node)
    graph.add_node("conversation", conversation_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        select_route,
        {
            "rag": "rag",
            "calculator": "calculator",
            "safety": "safety",
            "conversation": "conversation",
        },
    )

    graph.add_edge("rag", END)
    graph.add_edge("calculator", END)
    graph.add_edge("safety", END)
    graph.add_edge("conversation", END)

    return graph.compile()
