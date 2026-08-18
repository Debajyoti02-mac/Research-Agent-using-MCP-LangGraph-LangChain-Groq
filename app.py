"""
FastAPI web server for the Research Agent.
Wraps Agent_Server.py tools into a web chat interface.
Deployable free on Render / Railway / Koyeb.
"""

from __future__ import annotations

import os
import re
import uuid
import warnings
from typing import Optional, List

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
    StructuredTool.from_function(func=_file_read, name="read_file", description="Reads the text content of a local file on the filesystem. Example: 'notes.txt'"),
    StructuredTool.from_function(func=_write_file, name="write_file", description=_write_file.__doc__),
    StructuredTool.from_function(func=_write_file, name="file_write", description="Creates or updates a local file on the filesystem with the provided text content."),
    StructuredTool.from_function(func=_hybrid_rag, name="Hybrid_Rag", description=_hybrid_rag.__doc__),
    StructuredTool.from_function(func=_research_tool, name="Research_tool", description=_research_tool.__doc__),
]


# ──────────────────────────────────────────────
# LangGraph ReAct Agent (mirrors Client.py logic)
# ──────────────────────────────────────────────

memory = MemorySaver()

SYSTEM_PROMPT = (
    "You are a helpful and concise Research AI Assistant built with MCP, LangGraph, LangChain & Groq.\n"
    "- For calculations and arithmetic, use the calculator tool.\n"
    "- For research papers and document queries, use Research_tool or Hybrid_Rag.\n"
    "- For local files, use file_read to read files and write_file to save files.\n"
    "- For weather, use weather.\n"
    "Provide direct, accurate, and clean answers without repetition."
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


def clean_and_deduplicate(text: str, calc_output: str = None) -> str:
    """Sanitizes text, enforces bare numerical answers for calculations, and removes redundant repetitions."""
    if calc_output:
        return str(calc_output).strip()
    if not text:
        return ""
    text = text.strip()

    # Extract bare number if response is a calculation expression e.g. (12 \times 8 + 9 = 105) or 12 * 8 = 96 or **105**
    eq_match = re.search(r'=\s*\*?\*?([0-9\.\-]+)\*?\*?\s*\)?\.?$', text)
    if eq_match:
        return eq_match.group(1).strip()

    res_match = re.search(r'(?:is|result is|answer is|equals)\s*\*?\*?([0-9\.\-]+)\*?\*?\.?$', text, flags=re.IGNORECASE)
    if res_match:
        return res_match.group(1).strip()

    bold_match = re.match(r'^\*{1,2}([0-9\.\-]+)\*{1,2}$', text)
    if bold_match:
        return bold_match.group(1).strip()

    # 1. Deduplicate consecutive identical lines/paragraphs
    lines = text.split("\n")
    cleaned_lines = []
    for l in lines:
        stripped = l.strip()
        if stripped and cleaned_lines and cleaned_lines[-1].strip() == stripped:
            continue
        cleaned_lines.append(l)
    text = "\n".join(cleaned_lines).strip()

    # 2. Fix repeated exact phrases or concatenated digits (e.g. '469469', '100 100')
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
            if getattr(msg, "name", "") == "calculator" or getattr(msg, "type", "") == "tool":
                if "calculator" in tools_used and getattr(msg, "content", None):
                    calculator_output = str(msg.content).strip()

        # If calculator was called, prioritize the exact calculator tool output
        if calculator_output and "calculator" in tools_used:
            final_response = calculator_output
        else:
            final_response = result["messages"][-1].content if result.get("messages") else ""

        final_response = clean_and_deduplicate(final_response, calculator_output)

        return ChatResponse(
            response=final_response,
            thread_id=thread_id,
            tools_used=tools_used,
        )

    except Exception as e:
        # Attempt recovery if Groq failed on tool JSON parsing
        import json
        try:
            error_str = str(e)
            match = re.search(r"'failed_generation':\s*'(.*?)'\s*\}\s*\}", error_str, re.DOTALL)
            if not match:
                match = re.search(r'(\{"name":\s*"write_file".*?\})', error_str, re.DOTALL)

            if match:
                raw_payload = match.group(1).replace("\\'", "'")
                sanitized = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', raw_payload)
                parsed = json.loads(sanitized)

                tool_name = parsed.get("name")
                args = parsed.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', args))

                if tool_name == "write_file":
                    res = write_file(args.get("filename", "output.txt"), args.get("content", ""))
                    return ChatResponse(response=res, thread_id=thread_id, tools_used=["write_file"])
                elif tool_name == "file_read":
                    res = file_read(args.get("filename", ""))
                    return ChatResponse(response=res, thread_id=thread_id, tools_used=["file_read"])
                elif tool_name == "calculator":
                    expr = args.get("expression") or args.get("exec") or ""
                    res = calculator(expr)
                    return ChatResponse(response=res, thread_id=thread_id, tools_used=["calculator"])
        except Exception:
            pass

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

    # Save copy to uploads/ directory for direct file_read tool access
    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", filename)
    with open(save_path, "wb") as sf:
        sf.write(contents)

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


@app.get("/files")
async def list_files():
    """List all available files in uploads directory."""
    os.makedirs("uploads", exist_ok=True)
    files = [f for f in os.listdir("uploads") if not f.startswith(".")]
    file_list = []
    for f in files:
        f_path = os.path.join("uploads", f)
        file_list.append({
            "name": f,
            "size": os.path.getsize(f_path) if os.path.isfile(f_path) else 0,
        })
    return {"files": file_list}


@app.get("/files/{filename}")
async def get_file(filename: str):
    """Download a file from uploads or current directory."""
    clean_name = os.path.basename(filename)
    upload_path = os.path.join("uploads", clean_name)
    if os.path.exists(upload_path) and os.path.isfile(upload_path):
        return FileResponse(upload_path, filename=clean_name)
    if os.path.exists(clean_name) and os.path.isfile(clean_name):
        return FileResponse(clean_name, filename=clean_name)
    raise HTTPException(404, "File not found")


@app.get("/health")
async def health():
    """Health check endpoint for deployment platforms."""
    return {"status": "ok", "agent": "Research Agent"}


# ── Entry point ──

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))