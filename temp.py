# .venv\Scripts\activate

from backend import (
    chatbot,
    get_all_threads,
    ingest_rag_document
)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

from langgraph.types import Command

import streamlit as st
import uuid
import tempfile
import os


# ============================================================================
# Page configuration (must be first Streamlit call)
# ============================================================================

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Custom CSS theme
# ============================================================================

st.markdown("""
<style>

/* ---------- Global ---------- */
.stApp {
    background: linear-gradient(180deg, #0f1117 0%, #14161f 100%);
}

/* Hide default Streamlit chrome we don't need */
#MainMenu, footer {visibility: hidden;}

/* ---------- Title ---------- */
.app-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.4rem 0 1.2rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.2rem;
}
.app-header .emoji {
    font-size: 2rem;
}
.app-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg, #7c9cff, #b57cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-header .subtitle {
    font-size: 0.85rem;
    color: #8a8fa3;
    margin-top: 0.1rem;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #14161f;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stButton button {
    width: 100%;
    text-align: left;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    color: #d7d9e3;
    border-radius: 10px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.4rem;
    font-size: 0.85rem;
    transition: all 0.15s ease;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(124,156,255,0.12);
    border-color: rgba(124,156,255,0.4);
    color: #ffffff;
}

/* "New Chat" button gets a highlight */
section[data-testid="stSidebar"] .stButton:first-of-type button {
    background: linear-gradient(90deg, #7c9cff, #b57cff);
    color: #10111a;
    font-weight: 600;
    border: none;
}
section[data-testid="stSidebar"] .stButton:first-of-type button:hover {
    filter: brightness(1.08);
}

/* ---------- Chat bubbles ---------- */
div[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 0.4rem 0.2rem;
    margin-bottom: 0.6rem;
}

/* ---------- Status / tool call boxes ---------- */
div[data-testid="stStatusWidget"] {
    border-radius: 10px;
    border: 1px solid rgba(124,156,255,0.25);
}

/* ---------- Warning box for HITL approval ---------- */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* ---------- Chat input ---------- */
div[data-testid="stChatInput"] {
    border-radius: 14px;
}

/* ---------- Buttons (approve/reject) ---------- */
.stButton button[kind="primary"] {
    background: linear-gradient(90deg, #34d399, #10b981);
    border: none;
}
.stButton button[kind="primary"]:hover {
    filter: brightness(1.08);
}

</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper functions
# ============================================================================

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["pending_hitl"] = None
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


# ---------------------------------------------------------------------------
# HITL helper functions
# ---------------------------------------------------------------------------

def get_pending_interrupt(thread_id):
    """Return the first unresolved LangGraph interrupt for a thread."""

    config = {"configurable": {"thread_id": thread_id}}

    try:
        state_snapshot = chatbot.get_state(config)

        direct_interrupts = getattr(state_snapshot, "interrupts", ()) or ()
        if direct_interrupts:
            return direct_interrupts[0]

        tasks = getattr(state_snapshot, "tasks", ()) or ()
        for task in tasks:
            task_interrupts = getattr(task, "interrupts", ()) or ()
            if task_interrupts:
                return task_interrupts[0]

    except Exception:
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):
    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:
        save_pending_interrupt(thread_id, pending_interrupt)
    else:
        current_pending = st.session_state.get("pending_hitl")
        if current_pending is not None and current_pending.get("thread_id") == thread_id:
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    pending_hitl = st.session_state.get("pending_hitl")

    if not pending_hitl:
        st.warning("There is no pending action to approve or reject.")
        return

    interrupted_thread_id = pending_hitl["thread_id"]

    resume_config = {
        "configurable": {"thread_id": interrupted_thread_id},
        "metadata": {"thread_id": interrupted_thread_id},
        "run_name": "hitl_resume_trace",
    }

    try:
        with st.chat_message("assistant", avatar="🤖"):

            status_holder = {
                "box": st.status("🔄 Resuming the requested action...", expanded=True)
            }

            def resumed_ai_only_stream():
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                    if isinstance(message_chunk, AIMessage):
                        if message_chunk.content:
                            yield message_chunk.content

            resumed_ai_message = st.write_stream(resumed_ai_only_stream())

            next_interrupt = get_pending_interrupt(interrupted_thread_id)

            if next_interrupt is not None:
                save_pending_interrupt(interrupted_thread_id, next_interrupt)
                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False
                )
            else:
                st.session_state["pending_hitl"] = None
                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False
                )

        if resumed_ai_message:
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": resumed_ai_message
            })

        st.rerun()

    except Exception as error:
        st.error(f"Could not resume the requested action: {error}")


# ============================================================================
# Session state initialization
# ============================================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

add_thread(st.session_state["thread_id"])

sync_pending_interrupt(st.session_state["thread_id"])


# ============================================================================
# Header
# ============================================================================

st.markdown("""
<div class="app-header">
    <div class="emoji">🤖</div>
    <div>
        <h1>Agentic Chatbot</h1>
        <div class="subtitle">Search &nbsp;•&nbsp; RAG &nbsp;•&nbsp; Stocks &nbsp;•&nbsp; Weather &nbsp;•&nbsp; Human-in-the-loop</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("### 💬 Conversations")

    if st.button("➕  New Chat"):
        reset_chat()
        st.rerun()

    st.markdown("<div style='margin: 0.6rem 0; opacity:0.5;'>—</div>", unsafe_allow_html=True)

    for thread_id in st.session_state["chat_threads"][::-1]:

        short_label = f"🗂️ {str(thread_id)[:8]}…"

        if st.button(short_label, key=thread_id):
            st.session_state["thread_id"] = thread_id

            messages = load_conversation(thread_id)
            temp_messages = []

            for message in messages:
                if isinstance(message, HumanMessage):
                    role = "user"
                elif isinstance(message, AIMessage):
                    role = "assistant"
                else:
                    continue

                temp_messages.append({
                    "role": role,
                    "content": message.content
                })

            st.session_state["message_history"] = temp_messages
            sync_pending_interrupt(thread_id)
            st.rerun()


# ============================================================================
# Main chat interface
# ============================================================================

for message in st.session_state["message_history"]:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# ---------------------------------------------------------------------------
# HITL approval interface
# ---------------------------------------------------------------------------

pending_hitl = st.session_state.get("pending_hitl")

current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id") == st.session_state["thread_id"]
)

if current_thread_has_pending_hitl:

    st.warning(
        f"🧑 **Human approval required**\n\n{pending_hitl['prompt']}"
    )

    approve_column, reject_column = st.columns(2)

    with approve_column:
        if st.button(
            "✅ Approve Purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True
        ):
            resume_hitl_execution("yes")

    with reject_column:
        if st.button(
            "❌ Reject Purchase",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True
        ):
            resume_hitl_execution("no")


# ============================================================================
# Fixed chat input with PDF upload
# ============================================================================

submission = st.chat_input(
    "Ask me anything, or attach a PDF…",
    accept_file=True,
    file_type=["pdf"],
    disabled=current_thread_has_pending_hitl
)

user_input = None

if submission:
    user_input = submission.text
    uploaded_files = submission.files

    if uploaded_files:
        uploaded_pdf = uploaded_files[0]
        temporary_file_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
                temporary_file.write(uploaded_pdf.getvalue())
                temporary_file_path = temporary_file.name

            with st.spinner(f"📄 Processing {uploaded_pdf.name}..."):
                ingest_rag_document(temporary_file_path)

            st.toast(f"{uploaded_pdf.name} processed successfully.", icon="✅")

        except Exception as error:
            st.error(f"PDF processing failed: {error}")

        finally:
            if temporary_file_path and os.path.exists(temporary_file_path):
                os.remove(temporary_file_path)


if user_input:

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_trace",
    }

    with st.chat_message("assistant", avatar="🤖"):

        status_holder = {"box": None}

        def ai_only_stream():

            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):

                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

            pending_interrupt = get_pending_interrupt(st.session_state["thread_id"])

            if pending_interrupt is not None:
                save_pending_interrupt(st.session_state["thread_id"], pending_interrupt)
                yield (
                    "\n\n⚠️ This stock purchase requires your approval. "
                    "Use the **Approve Purchase** or **Reject Purchase** "
                    "button below."
                )

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            if get_pending_interrupt(st.session_state["thread_id"]) is not None:
                status_holder["box"].update(
                    label="⏸️ Waiting for human approval",
                    state="complete",
                    expanded=False
                )
            else:
                status_holder["box"].update(
                    label="✅ Tool finished",
                    state="complete",
                    expanded=False
                )

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })

    if (
        st.session_state.get("pending_hitl") is not None
        and st.session_state["pending_hitl"].get("thread_id") == st.session_state["thread_id"]
    ):
        st.rerun()