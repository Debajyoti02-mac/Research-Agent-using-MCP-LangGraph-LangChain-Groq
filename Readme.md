# Research Agent using MCP, LangGraph, LangChain & Groq

A Research Agent built with **MCP (Model Context Protocol)**, **LangGraph**, **LangChain**, **Groq**, **ChromaDB**, **BM25**, and **Tavily Search**.

The agent can:

- Answer questions from local research papers
- Perform Hybrid RAG retrieval
- Fall back to web search using Tavily
- Read files
- Write files
- Check weather
- Perform calculations
- Maintain conversation memory
- Trace executions using LangSmith

---

# Architecture

```text
User
  │
  ▼
LangGraph ReAct Agent
  │
  ▼
MCP Client
  │
  ▼
MCP Server
  │
  ├── Calculator Tool
  ├── Weather Tool
  ├── File Read Tool
  ├── File Write Tool
  ├── Hybrid_Rag Tool
  └── Research_tool
  │
  ▼
Groq LLM
```

---

# Tech Stack

- LangChain
- LangGraph
- LangSmith
- MCP
- ChromaDB
- Sentence Transformers
- BM25
- Tavily Search
- Groq
- Python

---

# Features

## 1. Calculator Tool

Performs arithmetic operations.

Example:

```python
calculator("25*10")
```

Output:

```text
250
```

---

## 2. Weather Tool

Fetches weather information.

Example:

```python
weather("Ranchi")
```

Output:

```text
Ranchi: 🌦 +29°C
```

---

## 3. File Read Tool

Reads file content.

Example:

```python
file_read("notes.txt")
```

---

## 4. File Write Tool

Creates and writes files.

Example:

```python
write_file(
    "MCP.txt",
    "Hello MCP"
)
```

---

## 5. Hybrid RAG

Combines:

- ChromaDB Vector Search
- BM25 Keyword Search
- Reciprocal Rank Fusion (RRF)

Flow:

```text
Question
   │
   ▼
Query Rewriter
   │
   ▼
Vector Search
   │
   ▼
BM25 Search
   │
   ▼
RRF Fusion
   │
   ▼
Top Documents
```

---

## 6. Tavily Fallback

If no relevant chunks are found:

```text
Hybrid_Rag
      │
      ▼
No Documents Found
      │
      ▼
Tavily Search
      │
      ▼
Web Results
```

---

## 7. Research Tool

Uses retrieved context to generate a final answer.

Flow:

```text
User Question
      │
      ▼
Hybrid_Rag
      │
      ▼
Retrieved Context
      │
      ▼
Groq LLM
      │
      ▼
Final Answer
```

---

# Project Structure

```text
Research_Agent/
│
├── Agent_Server.py
├── Client.py
├── .env
├── Research_Agent/
│   └── chromadb files
│
├── 2005.11401v4.pdf
├── checkpoints.db
└── README.md
```

---

# Installation

Create Virtual Environment

```bash
python -m venv .venv
```

Activate Environment

Mac/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install Dependencies

```bash
pip install langchain
pip install langgraph
pip install langchain-groq
pip install langchain-community
pip install langchain-text-splitters
pip install langchain-mcp-adapters
pip install chromadb
pip install sentence-transformers
pip install rank-bm25
pip install tavily-python
pip install pypdf
pip install python-dotenv
pip install mcp
```

---

# Environment Variables

Create a `.env` file.

```env
GROQ_API_KEY=your_groq_api_key

TAVILY_API_KEY=your_tavily_api_key

LANGSMITH_TRACING=true
LANGSMITH_PROJECT=Langchain_LangGraph
LANGSMITH_API_KEY=your_langsmith_key
```

---

# Run MCP Server

```bash
python Agent_Server.py
```

---

# Run Client

```bash
python Client.py
```

---

# Memory

Uses LangGraph MemorySaver.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
```

Thread-based memory:

```python
config = {
    "configurable": {
        "thread_id": "research_002"
    }
}
```

---

# LangSmith Monitoring

Enable tracing:

```env
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=Langchain_LangGraph
LANGSMITH_API_KEY=your_key
```

View traces:

```text
https://smith.langchain.com
```

---

# Example Query

```text
What is Retrieval Augmented Generation?
```

Flow:

```text
Question
  │
  ▼
Research_tool
  │
  ▼
Hybrid_Rag
  │
  ├── ChromaDB
  ├── BM25
  └── Tavily
  │
  ▼
Groq LLM
  │
  ▼
Final Answer
```

---

# Future Improvements

- Source Citations
- Persistent SQLite Memory
- Cross Encoder Reranker
- Multi Query Retrieval
- Reflection Agent
- Multi Agent Workflow
- Human In The Loop
- Document Upload Tool
- Research Report Generator

---

# Author

Debajyoti Hazra

BCA Student | AI & GenAI Enthusiast