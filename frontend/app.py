
import os
import time
from uuid import uuid4

import requests
import streamlit as st


API_URL = os.getenv(
    "FINAGENT_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


st.set_page_config(
    page_title="FinAgent AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
    }
    [data-testid="stChatMessage"] {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def start_new_chat():
    st.session_state.session_id = str(uuid4())
    st.session_state.messages = []


if "session_id" not in st.session_state:
    start_new_chat()


def request_answer(question: str) -> dict:
    response = requests.post(
        f"{API_URL}/agent/conversation",
        json={
            "session_id": st.session_state.session_id,
            "question": question,
        },
        timeout=180,
    )

    response.raise_for_status()
    return response.json()


def check_backend() -> bool:
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


# ----------------------------
# Sidebar
# ----------------------------

with st.sidebar:
    st.title("🏦 FinAgent AI")
    st.caption("Banking Intelligence Assistant")

    st.divider()

    if st.button(
        "➕ New Conversation",
        use_container_width=True,
    ):
        start_new_chat()
        st.rerun()

    if check_backend():
        st.success("Backend Connected")
    else:
        st.error("Backend Offline")

    st.divider()

    st.subheader("AI Capabilities")
    st.markdown(
        """
        - Conversational AI
        - Retrieval-Augmented Generation
        - LangGraph Agent Routing
        - Secure Demo Calculations
        - Context-Aware Conversation
        """
    )

    st.divider()

    st.caption(
        "Independent portfolio project. "
        "Not affiliated with any bank."
    )


# ----------------------------
# Main Chat Interface
# ----------------------------

st.title("FinAgent AI")
st.caption(
    "Secure Banking Conversational AI Assistant"
)

st.info(
    "Demo only. Uses fictional banking documents. "
    "Do not enter personal banking information, "
    "passwords, PINs or verification codes."
)


if not st.session_state.messages:
    st.markdown("### What can I help you with?")

    st.markdown(
        """
        Try asking:

        - How can I open a savings account?
        - What should I do if my credit card is lost?
        - What is the interest rate of the demo savings account?
        - calculate: 250 * 1.15
        """
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])
            route = message.get("route", "unknown")
            latency = message.get("latency")

            with st.expander("Response details"):
                st.write(f"Agent route: `{route}`")

                if sources:
                    st.write("Retrieved sources:")
                    for source in sources:
                        st.code(source)
                else:
                    st.write("No document sources.")

                if latency is not None:
                    st.write(
                        f"Response time: {latency:.2f} seconds"
                    )


question = st.chat_input(
    "Ask a banking question...",
    max_chars=2000,
)


if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("FinAgent AI is thinking..."):
            started = time.perf_counter()

            try:
                data = request_answer(question)

                latency = time.perf_counter() - started

                answer = data["answer"]
                sources = data.get("sources", [])
                route = data.get("route", "unknown")

                st.markdown(answer)

                with st.expander("Response details"):
                    st.write(f"Agent route: `{route}`")

                    if sources:
                        st.write("Retrieved sources:")
                        for source in sources:
                            st.code(source)
                    else:
                        st.write("No document sources.")

                    st.write(
                        f"Response time: {latency:.2f} seconds"
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "route": route,
                    "latency": latency,
                })

            except requests.Timeout:
                st.error(
                    "The AI service timed out. "
                    "Please try again."
                )

            except requests.RequestException:
                st.error(
                    "Cannot reach the AI service. "
                    "Check that FastAPI is running."
                )

            except (KeyError, ValueError):
                st.error(
                    "The AI service returned an "
                    "unexpected response."
                )
