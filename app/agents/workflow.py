
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


def create_agent_workflow(rag_pipeline):
    def router_node(state: AgentState) -> dict:
        question = state["question"].strip().lower()

        if question.startswith("calculate:"):
            route = "calculator"
        elif any(term in question for term in BANKING_TERMS):
            route = "rag"
        else:
            route = "safety"

        return {"route": route}

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

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        select_route,
        {
            "rag": "rag",
            "calculator": "calculator",
            "safety": "safety",
        },
    )

    graph.add_edge("rag", END)
    graph.add_edge("calculator", END)
    graph.add_edge("safety", END)

    return graph.compile()
