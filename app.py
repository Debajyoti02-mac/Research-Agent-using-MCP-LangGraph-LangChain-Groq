"""
FastAPI web server for the Research Agent.
Wraps Agent_Server.py tools into a web chat interface.
Deployable free on Render / Railway / Koyeb.
"""

import os
import re
import uuid
import warnings

# Set working directory so Agent_Server.py can find its relative paths (PDF, ChromaDB)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ──────────────────────────────────────────────
# Import tools & LLM getter from Agent_Server.py
# NOTE: import get_llm (function), not LLM (variable) —
# LLM is a lazy global that's None until get_llm() is called.
# ──────────────────────────────────────────────
from Agent_Server import calculator, weather, file_read, write_file, Hybrid_Rag, Research_tool, get_llm
import Agent_Server  # Module-level access for live updates (collection, etc.)

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tempfile


# ──────────────────────────────────────────────
# LangChain Tool Wrappers
# ──────────────────────────────────────────────

def _calculator(expression: str) -> str:
    """Evaluates a mathematical expression and returns the exact numerical answer. Example input: '25 * 10 + 5'. Output only the direct answer."""
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
    """Searches loaded research papers using retrieval (ChromaDB vectors). Returns the most relevant document chunks."""
    return Hybrid_Rag(query)


def _research_tool(query: str) -> str:
    """Primary tool for research questions. Retrieves paper context via RAG and generates a comprehensive LLM answer. Falls back to Tavily web search if no relevant papers are found."""
    return Research_tool(query)


tools = [
    StructuredTool.from_function(func=_calculator, name="calculator", description=_calculator.__doc__),
    StructuredTool.from_function(func=_weather, name="weather", description=_weather.__doc__),
    StructuredTool.from_function(func=_file_read, name="file_read", description=_file_read.__doc__),
    StructuredTool.from_function(func=_write_file, name="write_file", description=_write_file.__doc__),
    StructuredTool.from_function(func=_hybrid_rag, name="Hybrid_Rag", description=_hybrid_rag.__doc__),
    StructuredTool.from_function(func=_research_tool, name="Research_tool", description=_research_tool.__doc__),
]


# ──────────────────────────────────────────────
# LangGraph ReAct Agent (mirrors Client.py logic)
# ──────────────────────────────────────────────

memory = MemorySaver()

SYSTEM_PROMPT = (
    "You are a precise, concise Research AI Assistant built with MCP, LangGraph, LangChain & Groq.\n"
    "CRITICAL RULES:\n"
    "1. Never repeat your answers, restate previous messages, or duplicate information.\n"
    "2. For mathematical calculations & large numbers: ALWAYS use the calculator tool. Reply with ONLY the single final answer directly without extra text, conversation, steps, or repeating long digit sequences.\n"
    "3. For research / paper questions: Use Research_tool or Hybrid_Rag. Give a clear, direct summary without repetitive sentences.\n"
    "4. For local file operations: Use file_read or write_file.\n"
    "5. For weather: Use weather.\n"
    "6. Keep answers clean, accurate, and non-repetitive."
)

agent = create_react_agent(
    model=get_llm(),
    tools=tools,
    prompt=SYSTEM_PROMPT,
    checkpointer=memory,
)


# ──────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────

app = FastAPI(
    title="Research Agent",
    description="AI Research Assistant powered by MCP, LangGraph, LangChain & Groq",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # wildcard origin + credentials is invalid per CORS spec
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend assets
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / Response models ──

class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    tools_used: list[str]


# ── Routes ──

@app.get("/")
async def root():
    """Serve the chat frontend."""
    return FileResponse("static/index.html")


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


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Process a user message through the ReAct agent."""
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await agent.ainvoke(
            {"messages": [("user", req.message)]},
            config=config,
        )

        # Collect tool names called in the current turn (since last user message)
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
            # Capture tool message content
            if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == "calculator":
                calculator_output = str(getattr(msg, "content", "")).strip()

        # If only calculator was called, prioritize the exact calculator tool output
        if tools_used == ["calculator"] and calculator_output:
            final_response = calculator_output
        else:
            final_response = result["messages"][-1].content if result.get("messages") else ""

        final_response = clean_and_deduplicate(final_response)

        return ChatResponse(
            response=final_response,
            thread_id=thread_id,
            tools_used=tools_used,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or text document to the agent's knowledge base."""
    filename = file.filename or "document"
    suffix = os.path.splitext(filename)[1].lower()

    if suffix not in (".pdf", ".txt", ".md", ".csv"):
        raise HTTPException(400, "Only PDF, TXT, MD, and CSV files are supported.")

    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 15 MB).")

    # Save to temp file for loaders
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(contents)
        tmp.close()

        # Ensure ChromaDB collection is initialized before we touch it —
        # collection is None until Agent_Server.initialize_rag() has run once.
        Agent_Server.initialize_rag()

        # Load document
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp.name)
            pages = loader.load()
        else:
            text_content = contents.decode("utf-8", errors="ignore")
            pages = [Document(page_content=text_content, metadata={"source": filename})]

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_overlap=180, chunk_size=1200)
        texts = splitter.split_documents(pages)
        new_chunks = [t.page_content for t in texts]
        new_metadata = [{**t.metadata, "source": filename} for t in texts]

        if not new_chunks:
            raise HTTPException(400, "No text content could be extracted from the file.")

        # Add to ChromaDB
        start_id = Agent_Server.collection.count()
        Agent_Server.collection.add(
            ids=[str(i) for i in range(start_id, start_id + len(new_chunks))],
            documents=new_chunks,
            metadatas=new_metadata,
        )

        return {
            "status": "success",
            "filename": filename,
            "chunks_added": len(new_chunks),
            "total_chunks": Agent_Server.collection.count(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to process document: {e}")
    finally:
        os.unlink(tmp.name)


@app.get("/health")
async def health():
    """Health check endpoint for deployment platforms."""
    return {"status": "ok", "agent": "Research Agent"}


# ── Entry point ──

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))