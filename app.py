"""
FastAPI web server for the Research Agent.
Wraps Agent_Server.py tools into a web chat interface.
Deployable free on Render / Railway / Koyeb.
"""

import os
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
# Import tools & LLM from Agent_Server.py
# This triggers module-level init: PDF loading, ChromaDB, BM25, LLM creation
# Takes ~10-20 s on first cold start
# ──────────────────────────────────────────────
from Agent_Server import calculator, weather, file_read, write_file, Hybrid_Rag, Research_tool, LLM
import Agent_Server  # Module-level access for live updates (chunks, collection, BM25)

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import tempfile


# ──────────────────────────────────────────────
# LangChain Tool Wrappers
# ──────────────────────────────────────────────

def _calculator(expression: str) -> str:
    """Evaluates a mathematical expression. Example input: '25 * 10 + 5'"""
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
    """Searches loaded research papers using hybrid retrieval (ChromaDB vectors + BM25 keywords + Reciprocal Rank Fusion). Returns the most relevant document chunks."""
    return Hybrid_Rag(query)


def _research_tool(query: str) -> str:
    """Primary tool for research questions. Retrieves paper context via Hybrid RAG and generates a comprehensive LLM answer. Falls back to Tavily web search if no relevant papers are found."""
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
    "You are a Research AI Assistant built with MCP, LangGraph, LangChain & Groq.\n"
    "For any research or paper-related questions, always use Research_tool first.\n"
    "For raw document retrieval, use Hybrid_Rag.\n"
    "For reading files from the file system, use file_read.\n"
    "For writing or saving notes/files to the file system, use write_file.\n"
    "For math, use calculator. For weather, use weather.\n"
    "Provide clear, well-structured answers. Use markdown formatting when helpful."
)

agent = create_react_agent(
    model=LLM,
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
    allow_credentials=True,
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

        # Collect tool names the agent actually called
        tools_used = []
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "")
                    if name and name not in tools_used:
                        tools_used.append(name)

        return ChatResponse(
            response=result["messages"][-1].content,
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

        # Update in-memory chunks list & rebuild BM25 index
        Agent_Server.chunks.extend(new_chunks)
        new_tokens = [c.split() for c in Agent_Server.chunks]
        Agent_Server.token_cor = BM25Okapi(new_tokens)

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
