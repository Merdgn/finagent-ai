
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

from frontend.cloud_service import get_cloud_service

load_dotenv(PROJECT_ROOT / ".env")


# Use Groq for the public cloud demo.
os.environ["LLM_PROVIDER"] = "groq"

# Streamlit Cloud secrets are configured in
# the application's deployment settings.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets[
            "GROQ_API_KEY"
        ]
except FileNotFoundError:
    pass

os.environ["GROQ_MODEL"] = "openai/gpt-oss-20b"


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



@st.cache_resource(show_spinner=False)
def load_ai_service():
    return get_cloud_service()


def request_answer(question: str) -> dict:
    service = load_ai_service()

    result = service.ask(
        session_id=st.session_state.session_id,
        question=question,
    )

    return {
        "answer": result.answer,
        "route": result.route,
        "sources": result.sources,
        "standalone_question": result.standalone_question,
    }


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

    st.caption("Cloud AI Demo")
    st.success("Groq-powered AI")

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

            except Exception:
                st.error(
                    "The AI service is temporarily unavailable. "
                    "Please try again later."
                )

            except (KeyError, ValueError):
                st.error(
                    "The AI service returned an "
                    "unexpected response."
                )
