"""
Streamlit Web Application for the Research Agent.
Interactive UI powered by MCP, LangGraph, LangChain & Groq.
"""

import os
import re
import uuid
import tempfile
import warnings
import streamlit as st

# Set working directory so Agent_Server can resolve relative paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

# Sync Streamlit Cloud secrets to os.environ if available
try:
    if hasattr(st, "secrets"):
        for key, val in st.secrets.items():
            if isinstance(val, str):
                os.environ.setdefault(key, val)
except Exception:
    pass

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import Agent_Server
from Agent_Server import (
    calculator,
    weather,
    file_read,
    write_file,
    Hybrid_Rag,
    Research_tool,
    get_llm,
)


# ──────────────────────────────────────────────
# Page Configuration & Styling
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Styles */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Header styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    
    /* Tool Badge */
    .tool-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(139, 92, 246, 0.15);
        color: #c084fc;
        border: 1px solid rgba(139, 92, 246, 0.35);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 6px;
        margin-right: 4px;
    }
    
    /* Sidebar card styling */
    .sidebar-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Tools & Agent Initialization
# ──────────────────────────────────────────────

def _calculator(expression: str) -> str:
    """Evaluates a mathematical expression and returns the exact numerical answer. Example input: '25 * 10 + 5'."""
    return calculator(expression)


def _weather(location: str) -> str:
    """Fetches current weather for a given city or location name."""
    return weather(location)


def _file_read(filename: str) -> str:
    """Reads the text content of a local file on the filesystem. Example: 'notes.txt'"""
    return file_read(filename)


def _write_file(filename: str, content: str) -> str:
    """Creates or updates a local file on the filesystem with the provided text content."""
    return write_file(filename, content)


def _hybrid_rag(query: str) -> str:
    """Searches loaded research papers using retrieval (ChromaDB vectors). Returns relevant document chunks."""
    return Hybrid_Rag(query)


def _research_tool(query: str) -> str:
    """Primary tool for research questions. Retrieves paper context via RAG and generates a comprehensive answer."""
    return Research_tool(query)


tools = [
    StructuredTool.from_function(func=_calculator, name="calculator", description=_calculator.__doc__),
    StructuredTool.from_function(func=_weather, name="weather", description=_weather.__doc__),
    StructuredTool.from_function(func=_file_read, name="file_read", description=_file_read.__doc__),
    StructuredTool.from_function(func=_file_read, name="read_file", description="Reads the text content of a local file on the filesystem. Example: 'notes.txt'"),
    StructuredTool.from_function(func=_write_file, name="write_file", description=_write_file.__doc__),
    StructuredTool.from_function(func=_write_file, name="file_write", description="Creates or updates a local file on the filesystem with the provided text content."),
    StructuredTool.from_function(func=_hybrid_rag, name="Hybrid_Rag", description=_hybrid_rag.__doc__),
    StructuredTool.from_function(func=_research_tool, name="Research_tool", description=_research_tool.__doc__),
]

SYSTEM_PROMPT = (
    "You are a precise, concise Research AI Assistant built with MCP, LangGraph, LangChain & Groq.\n"
    "CRITICAL RULES:\n"
    "1. Never repeat your answers, restate previous messages, or duplicate information.\n"
    "2. For mathematical calculations & numbers: ALWAYS use the calculator tool. Reply with ONLY the single final numerical answer directly without extra text, conversation, steps, or repeating long digit sequences.\n"
    "3. For research / paper questions: Use Research_tool or Hybrid_Rag. Give a clear, direct summary without repetitive sentences.\n"
    "4. For local file operations: Use file_read or write_file. When writing files, format plain text with clean newlines.\n"
    "5. For weather: Use weather.\n"
    "6. Keep answers clean, accurate, and non-repetitive."
)


@st.cache_resource
def get_agent():
    memory = MemorySaver()
    return create_react_agent(
        model=get_llm(),
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )


agent = get_agent()


# ──────────────────────────────────────────────
# Helper: Tool Recovery & Sanitization
# ──────────────────────────────────────────────

TOOL_ICONS = {
    "calculator": "🧮",
    "weather": "🌤️",
    "file_read": "📖",
    "write_file": "✍️",
    "Hybrid_Rag": "📄",
    "Research_tool": "🔬",
}

def clean_and_deduplicate(text: str) -> str:
    """Sanitizes text and removes redundant consecutive repetitions generated by LLMs."""
    if not text:
        return ""
    text = text.strip()

    # 1. Deduplicate consecutive identical lines/paragraphs
    lines = text.split("\n")
    cleaned_lines = []
    for l in lines:
        stripped = l.strip()
        if stripped and cleaned_lines and cleaned_lines[-1].strip() == stripped:
            continue
        cleaned_lines.append(l)
    text = "\n".join(cleaned_lines).strip()

    # 2. Fix repeated exact phrases or concatenated digits (e.g. '469469', '100 100', 'Answer: 100 Answer: 100')
    match = re.match(r"^(.+?)(?:[\s,]*\1)+$", text, flags=re.DOTALL)
    if match and len(match.group(1)) >= 1:
        text = match.group(1).strip()

    return text


def handle_tool_recovery(error_str: str):
    """
    Recovers from Groq's 400 'Failed to parse tool call arguments as JSON' error
    by sanitizing invalid escape codes and executing the intended tool directly.
    """
    import json
    try:
        # Match failed_generation payload
        match = re.search(r"'failed_generation':\s*'(.*?)'\s*\}\s*\}", error_str, re.DOTALL)
        if not match:
            match = re.search(r'(\{"name":\s*"write_file".*?\})', error_str, re.DOTALL)

        if match:
            raw_payload = match.group(1)
            # Unescape double-escaped quotes and newlines if present
            raw_payload = raw_payload.replace("\\'", "'")
            # Sanitize invalid escape sequences (e.g. \s, \c) into valid JSON escapes
            sanitized = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', raw_payload)
            parsed = json.loads(sanitized)

            tool_name = parsed.get("name")
            args = parsed.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', args))

            if tool_name == "write_file":
                filename = args.get("filename", "output.txt")
                content = args.get("content", "")
                res = write_file(filename, content)
                return res, ["write_file"]
            elif tool_name == "file_read":
                filename = args.get("filename", "")
                res = file_read(filename)
                return res, ["file_read"]
            elif tool_name == "calculator":
                expr = args.get("expression") or args.get("exec") or ""
                res = calculator(expr)
                return res, ["calculator"]
    except Exception:
        pass
    return None, []


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ──────────────────────────────────────────────
# Sidebar: Document Management & Config
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔬 Research Agent")
    st.markdown(
        """
        **Powered by:**
        - **Groq LLM** (Ultra-fast inference)
        - **LangGraph** (ReAct Agent Framework)
        - **MCP** (Model Context Protocol)
        - **Hybrid RAG** (ChromaDB + BM25)
        """
    )
    st.divider()

    st.markdown("### 📄 Document Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload PDF or TXT to Index",
        type=["pdf", "txt", "md", "csv"],
        help="Attached documents are automatically chunked and indexed into ChromaDB vectors for Hybrid RAG.",
    )

    if uploaded_file is not None:
        if st.button("📥 Index Uploaded Document", use_container_width=True):
            with st.spinner("Processing & indexing document chunks into ChromaDB..."):
                try:
                    filename = uploaded_file.name
                    suffix = os.path.splitext(filename)[1].lower()
                    contents = uploaded_file.read()

                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(contents)
                        tmp_path = tmp.name

                    Agent_Server.initialize_rag()

                    if suffix == ".pdf":
                        loader = PyPDFLoader(tmp_path)
                        pages = loader.load()
                    else:
                        text_content = contents.decode("utf-8", errors="ignore")
                        pages = [Document(page_content=text_content, metadata={"source": filename})]

                    splitter = RecursiveCharacterTextSplitter(chunk_overlap=180, chunk_size=1200)
                    texts = splitter.split_documents(pages)
                    new_chunks = [t.page_content for t in texts]
                    new_metadata = [{**t.metadata, "source": filename} for t in texts]

                    if new_chunks:
                        start_id = Agent_Server.collection.count()
                        Agent_Server.collection.add(
                            ids=[str(i) for i in range(start_id, start_id + len(new_chunks))],
                            documents=new_chunks,
                            metadatas=new_metadata,
                        )
                        st.success(f"✓ Indexed **{len(new_chunks)}** chunks from `{filename}` successfully!")
                    else:
                        st.warning("No readable text found in document.")

                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                except Exception as e:
                    st.error(f"Upload failed: {e}")

    try:
        Agent_Server.initialize_rag()
        total_chunks = Agent_Server.collection.count() if Agent_Server.collection else 0
        st.info(f"📊 **ChromaDB Index:** {total_chunks} active chunks")
    except Exception:
        pass

    st.divider()

    if st.button("🧹 Clear Chat / New Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()


# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────

st.markdown('<div class="main-title">Research Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask research questions, run math calculations, query documents, and manage files.</div>', unsafe_allow_html=True)

# Suggestion Chips if no messages yet
if not st.session_state.messages:
    st.markdown("##### 💡 Suggested Questions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 What is Retrieval Augmented Generation?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What is Retrieval Augmented Generation?", "tools": []})
            st.rerun()
        if st.button("🧮 Calculate 256 * 48 + 1024", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Calculate 256 * 48 + 1024", "tools": []})
            st.rerun()
    with col2:
        if st.button("🔬 Explain the RAG pipeline architecture", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Explain the RAG pipeline architecture", "tools": []})
            st.rerun()
        if st.button("🌤️ What is the weather in Ranchi?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What is the weather in Ranchi?", "tools": []})
            st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools"):
            badges_html = "".join(
                [f'<span class="tool-badge">{TOOL_ICONS.get(t, "⚙️")} {t}</span>' for t in msg["tools"]]
            )
            st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)

# User input box
user_input = st.chat_input("Ask a question about your research or enter a calculation…")

# Check if last message needs AI response (e.g. from suggestion buttons or chat input)
pending_prompt = None
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "tools": []})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    pending_prompt = st.session_state.messages[-1]["content"]

if pending_prompt:
    with st.chat_message("assistant"):
        with st.spinner("Thinking & processing tools..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            try:
                result = agent.invoke(
                    {"messages": [("user", pending_prompt)]},
                    config=config,
                )

                messages = result.get("messages", [])
                tools_used = []
                calculator_output = None
                last_user_idx = 0

                for idx, msg in enumerate(messages):
                    msg_type = getattr(msg, "type", "")
                    if msg_type in ("human", "user") or (isinstance(msg, tuple) and msg[0] == "user"):
                        last_user_idx = idx

                for msg in messages[last_user_idx:]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            name = tc.get("name", "")
                            if name and name not in tools_used:
                                tools_used.append(name)
                    if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == "calculator":
                        calculator_output = str(getattr(msg, "content", "")).strip()

                if tools_used == ["calculator"] and calculator_output:
                    final_response = calculator_output
                else:
                    final_response = result["messages"][-1].content if result.get("messages") else ""

                final_response = clean_and_deduplicate(final_response)

                st.markdown(final_response)
                if tools_used:
                    badges_html = "".join(
                        [f'<span class="tool-badge">{TOOL_ICONS.get(t, "⚙️")} {t}</span>' for t in tools_used]
                    )
                    st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_response,
                        "tools": tools_used,
                    }
                )

            except Exception as e:
                # Attempt recovery if Groq failed on tool JSON parsing
                recovered_resp, recovered_tools = handle_tool_recovery(str(e))
                if recovered_resp:
                    st.markdown(recovered_resp)
                    if recovered_tools:
                        badges_html = "".join(
                            [f'<span class="tool-badge">{TOOL_ICONS.get(t, "⚙️")} {t}</span>' for t in recovered_tools]
                        )
                        st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": recovered_resp,
                            "tools": recovered_tools,
                        }
                    )
                else:
                    st.error(f"Error generating response: {e}")
