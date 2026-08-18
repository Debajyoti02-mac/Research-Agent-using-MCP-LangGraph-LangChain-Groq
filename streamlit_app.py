"""
Streamlit Web Application for the Research Agent.
Interactive UI powered by MCP, LangGraph, LangChain & Groq.
Designed with a Pure OLED Black luxury aesthetic, masked API credentials, and modern typography.
"""

from __future__ import annotations

import os
import re
import uuid
import tempfile
import warnings
import streamlit as st
from typing import Optional, List

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
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Pure OLED Black Luxury Theme & Google Font
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Pure Black App Background */
    .stApp {
        background-color: #000000 !important;
        color: #f1f5f9 !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #050507 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* Hero Header */
    .hero-container {
        background: rgba(14, 15, 22, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 22px;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
    }
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #c084fc 0%, #60a5fa 50%, #2dd4bf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Cards */
    .status-card {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }

    /* Tool Badge */
    .tool-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(168, 85, 247, 0.14);
        color: #d8b4fe;
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 0.76rem;
        font-weight: 600;
        margin-top: 8px;
        margin-right: 6px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Code blocks */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Tools & Agent Initialization
# ──────────────────────────────────────────────

def _calculator(expression: str = "", exec: str = "", expr: str = "", **kwargs) -> str:
    """Evaluates a mathematical expression and returns the exact numerical answer. Example input: '25 * 10 + 5'."""
    return calculator(expression=expression, exec=exec, expr=expr, **kwargs)


def _weather(location: str) -> str:
    """Fetches current weather for a given city or location name."""
    return weather(location)


def _file_read(filename: str = "", file_path: str = "", path: str = "", **kwargs) -> str:
    """Reads the text content of a local file or PDF on the filesystem. Example: 'notes.txt'"""
    return file_read(filename=filename, file_path=file_path, path=path, **kwargs)


def _write_file(filename: str = "", content: str = "", file_path: str = "", text: str = "", **kwargs) -> str:
    """Creates or updates a local file on the filesystem with the provided text content."""
    return write_file(filename=filename, content=content, file_path=file_path, text=text, **kwargs)


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
    "You are a helpful, accurate, and concise Research AI Assistant built with MCP, LangGraph, LangChain & Groq.\n"
    "- For calculations and arithmetic: ALWAYS use the calculator tool. Reply with ONLY the final numerical answer directly without equations, formulas, LaTeX, steps, or conversation.\n"
    "- For uploaded documents, PDFs, or research questions: use file_read or Hybrid_Rag to read and analyze document content.\n"
    "- For local files: use file_read to read files and write_file to save files.\n"
    "- For weather: use weather.\n"
    "Provide direct, accurate, and clean answers without repetition."
)


def get_agent(api_key: str):
    if "agent" not in st.session_state or st.session_state.get("cached_api_key") != api_key:
        memory = MemorySaver()
        st.session_state["agent"] = create_react_agent(
            model=get_llm(groq_api_key=api_key),
            tools=tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=memory,
        )
        st.session_state["cached_api_key"] = api_key
    return st.session_state["agent"]


# ──────────────────────────────────────────────
# Helper: Tool Recovery & Sanitization
# ──────────────────────────────────────────────

TOOL_ICONS = {
    "calculator": "🧮",
    "weather": "🌤️",
    "file_read": "📖",
    "read_file": "📖",
    "write_file": "✍️",
    "file_write": "✍️",
    "Hybrid_Rag": "📄",
    "Research_tool": "🔬",
}


def clean_and_deduplicate(text: str, calc_output: str = None) -> str:
    """Sanitizes text, enforces bare numerical answers for calculations, and removes redundant repetitions."""
    if calc_output:
        return str(calc_output).strip()
    if not text:
        return ""
    text = text.strip()

    eq_match = re.search(r'=\s*\*?\*?([0-9\.\-]+)\*?\*?\s*\)?\.?$', text)
    if eq_match:
        return eq_match.group(1).strip()

    res_match = re.search(r'(?:is|result is|answer is|equals)\s*\*?\*?([0-9\.\-]+)\*?\*?\.?$', text, flags=re.IGNORECASE)
    if res_match:
        return res_match.group(1).strip()

    bold_match = re.match(r'^\*{1,2}([0-9\.\-]+)\*{1,2}$', text)
    if bold_match:
        return bold_match.group(1).strip()

    # Deduplicate consecutive identical lines
    lines = text.split("\n")
    cleaned_lines = []
    for l in lines:
        stripped = l.strip()
        if stripped and cleaned_lines and cleaned_lines[-1].strip() == stripped:
            continue
        cleaned_lines.append(l)
    text = "\n".join(cleaned_lines).strip()

    # Fix repeated exact phrases
    match = re.match(r"^(.+?)(?:[\s,]*\1)+$", text, flags=re.DOTALL)
    if match and len(match.group(1)) >= 1:
        text = match.group(1).strip()

    return text


def handle_tool_recovery(error_str: str):
    """Recovers from Groq's 400 JSON parse errors on tool payloads."""
    import json
    try:
        match = re.search(r"'failed_generation':\s*'(.*?)'\s*\}\s*\}", error_str, re.DOTALL)
        if not match:
            match = re.search(r'(\{"name":\s*"(?:write_file|file_write)".*?\})', error_str, re.DOTALL)

        if match:
            raw_payload = match.group(1).replace("\\'", "'")
            sanitized = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', raw_payload)
            parsed = json.loads(sanitized)

            tool_name = parsed.get("name")
            args = parsed.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', args))

            if tool_name in ("write_file", "file_write"):
                filename = args.get("filename") or args.get("file_path") or "output.txt"
                content = args.get("content") or args.get("text") or ""
                res = write_file(filename=filename, content=content)
                return res, ["write_file"]
            elif tool_name in ("file_read", "read_file"):
                filename = args.get("filename") or args.get("file_path") or ""
                res = file_read(filename=filename)
                return res, ["file_read"]
            elif tool_name == "calculator":
                expr = args.get("expression") or args.get("exec") or ""
                res = calculator(expression=expr)
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
# Sidebar: Safe API Masking & Knowledge Base
# ──────────────────────────────────────────────

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = groq_key

with st.sidebar:
    st.markdown("### 🔬 Research Agent")
    st.markdown(
        """
        **Architecture Stack:**
        - **Groq LLM** (`openai/gpt-oss-120b`)
        - **LangGraph** (ReAct Agent Framework)
        - **MCP Protocol** (FastMCP Tools)
        - **Hybrid RAG** (ChromaDB Vector Retrieval)
        """
    )
    st.divider()

    # ── Masked API Key Configuration ──
    st.markdown("##### 🔑 API Key Configuration")
    if groq_key:
        st.success("🔒 **Groq API Key**: Loaded & Secured from `.env`")
        with st.expander("Change API Key"):
            new_key = st.text_input(
                "Enter New Key",
                type="password",
                placeholder="gsk_...",
                help="Override environment key for this session",
            )
            if new_key:
                groq_key = new_key
                os.environ["GROQ_API_KEY"] = new_key
                st.success("Custom key applied.")
    else:
        user_api_key = st.text_input(
            "Enter Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your free key at https://console.groq.com/keys",
        )
        if user_api_key:
            groq_key = user_api_key
            os.environ["GROQ_API_KEY"] = user_api_key

    st.divider()

    # ── Document Knowledge Base ──
    st.markdown("##### 📄 Document Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, MD, CSV",
        type=["pdf", "txt", "md", "csv"],
        help="Attached documents are chunked and indexed into ChromaDB vectors for Hybrid RAG.",
    )

    if uploaded_file is not None:
        file_key = f"indexed_{uploaded_file.name}_{uploaded_file.size}"
        if file_key not in st.session_state:
            st.session_state[file_key] = False

        if not st.session_state[file_key]:
            if st.button("📥 Index Document", use_container_width=True, type="primary"):
                with st.spinner("Processing & indexing into ChromaDB..."):
                    try:
                        filename = uploaded_file.name
                        suffix = os.path.splitext(filename)[1].lower()
                        # Save directly to the original storage (root project directory)
                        save_path = filename
                        with open(save_path, "wb") as sf:
                            sf.write(contents)

                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(contents)
                            tmp_path = tmp.name

                        Agent_Server.initialize_rag()

                        pages = []
                        if suffix == ".pdf":
                            try:
                                loader = PyPDFLoader(tmp_path)
                                pages = loader.load()
                            except Exception:
                                import pypdf
                                reader = pypdf.PdfReader(tmp_path)
                                for idx, p in enumerate(reader.pages):
                                    text = p.extract_text() or ""
                                    if text.strip():
                                        pages.append(Document(page_content=text, metadata={"source": filename, "page": idx + 1}))
                        else:
                            text_content = contents.decode("utf-8", errors="ignore")
                            pages = [Document(page_content=text_content, metadata={"source": filename})]

                        splitter = RecursiveCharacterTextSplitter(chunk_overlap=180, chunk_size=1200)
                        texts = splitter.split_documents(pages)
                        new_chunks = [t.page_content for t in texts]
                        new_metadata = [{**t.metadata, "source": filename} for t in texts]

                        if new_chunks and Agent_Server.collection is not None:
                            import time
                            ts = int(time.time())
                            chunk_ids = [f"{filename}_{i}_{ts}" for i in range(len(new_chunks))]
                            Agent_Server.collection.add(
                                ids=chunk_ids,
                                documents=new_chunks,
                                metadatas=new_metadata,
                            )
                            st.session_state[file_key] = True
                            st.success(f"✓ Indexed **{len(new_chunks)}** chunks from `{filename}`!")
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": f"📄 **Document Indexed**: I have successfully indexed `{filename}` ({len(new_chunks)} chunks). You can now ask questions or request summaries about this document!",
                                    "tools": ["Hybrid_Rag"],
                                }
                            )
                            st.rerun()
                        else:
                            st.warning("No readable text could be extracted from this file.")

                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                    except Exception as e:
                        st.error(f"Upload failed: {e}")
        else:
            st.success(f"✓ `{uploaded_file.name}` is indexed.")

    try:
        Agent_Server.initialize_rag()
        total_chunks = Agent_Server.collection.count() if Agent_Server.collection else 0
        st.caption(f"📊 **Knowledge Base:** {total_chunks} active chunks")
    except Exception:
        pass

    st.divider()

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()


# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-container">
        <div class="main-title">🔬 Research Agent</div>
        <div class="sub-title">AI Research Assistant powered by MCP, LangGraph, LangChain & Groq. Query papers, generate code & files, perform math, and inspect documents.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not groq_key:
    st.warning("⚠️ **Groq API Key Required**: Please enter your Groq API Key in the sidebar to activate the Research Agent.")
    st.info("Don't have an API key? Get one for free at [console.groq.com/keys](https://console.groq.com/keys).")
else:
    agent = get_agent(groq_key)

    # Suggestion Chips if chat is empty
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

    # User chat input
    user_input = st.chat_input("Ask a research question, enter a calculation, or request file operations…")

    # Check pending message
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
                        if getattr(msg, "name", "") == "calculator" or getattr(msg, "type", "") == "tool":
                            if "calculator" in tools_used and getattr(msg, "content", None):
                                calculator_output = str(msg.content).strip()

                    if calculator_output and "calculator" in tools_used:
                        final_response = calculator_output
                    else:
                        final_response = result["messages"][-1].content if result.get("messages") else ""

                    final_response = clean_and_deduplicate(final_response, calculator_output)

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
                    st.rerun()

                except Exception as e:
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
                        st.rerun()
                    else:
                        st.error(f"Error generating response: {e}")
