from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Research_Agent")

# =========================
# Calculator Tool
# =========================

import ast
import operator
import math

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "abs": abs,
    "round": round,
    "pow": math.pow,
    "factorial": math.factorial,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_node(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    elif isinstance(node, ast.Constant):
        return node.value

    elif isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)

        op_type = type(node.op)

        if op_type in _OPERATORS:
            return _OPERATORS[op_type](left, right)

        raise TypeError(f"Unsupported operator {op_type}")

    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand)

        op_type = type(node.op)

        if op_type in _OPERATORS:
            return _OPERATORS[op_type](operand)

        raise TypeError(f"Unsupported operator {op_type}")

    elif isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]

        raise NameError(f"Unknown constant: {node.id}")

    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in _FUNCTIONS:
                args = [_safe_eval_node(arg) for arg in node.args]
                return _FUNCTIONS[node.func.id](*args)

        raise NameError("Unsupported function")

    raise TypeError("Unsupported expression")


@mcp.tool()
def calculator(exec: str):
    """Safely evaluates arithmetic expressions."""
    try:
        clean_expr = exec.strip().replace("^", "**")
        tree = ast.parse(clean_expr, mode="eval")
        result = _safe_eval_node(tree)
        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"


# =========================
# Weather Tool
# =========================

import requests


@mcp.tool()
def weather(location: str):
    """Get weather information."""
    try:
        response = requests.get(
            f"https://wttr.in/{location}?format=3"
        ).text

        return response

    except Exception as e:
        return str(e)


# =========================
# File Tools
# =========================

import os


@mcp.tool()
def file_read(filename: str):
    """Read a local file."""

    try:
        with open(
            filename,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            return f.read()

    except Exception as e:
        return str(e)


@mcp.tool()
def write_file(filename: str, content: str):
    """
    Write text to file.
    Keep content under 5000 characters.
    """
    if len(content) > 5000:
        return "Content too large. Please split into smaller chunks."

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    return "File saved successfully."


# =========================
# Environment
# =========================

from dotenv import load_dotenv

load_dotenv()

# =========================
# Lazy Globals
# =========================

collection = None
token_cor = None
chunks = None
LLM = None


# =========================
# Lazy LLM
# =========================

def get_llm():
    global LLM

    if LLM is None:
        from langchain_groq import ChatGroq

        LLM = ChatGroq(
            model="openai/gpt-oss-120b"
        )

    return LLM


# =========================
# Lazy RAG Initialization
# =========================

def initialize_rag():

    global collection
    global token_cor
    global chunks

    if collection is not None:
        return

    print("Loading RAG resources...")

    from langchain_community.document_loaders import (
        PyPDFLoader
    )

    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter
    )

    import chromadb

    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction
    )

    from rank_bm25 import BM25Okapi

    loader = PyPDFLoader(
        "2005.11401v4.pdf"
    )

    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180
    )

    texts = splitter.split_documents(
        pages
    )

    chunks = [
        doc.page_content
        for doc in texts
    ]

    embedding_fun = (
        SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    client = chromadb.PersistentClient(
        path="./Research_Agent"
    )

    collection = client.get_or_create_collection(
        name="Agent",
        embedding_function=embedding_fun
    )

    if collection.count() == 0:

        collection.add(
            ids=[
                str(i)
                for i in range(len(chunks))
            ],
            documents=chunks,
            metadatas=[
                doc.metadata
                for doc in texts
            ]
        )

    token = [
        chunk.split()
        for chunk in chunks
    ]

    token_cor = BM25Okapi(token)

    print("RAG initialized successfully")


# =========================
# Hybrid RAG
# =========================

@mcp.tool()
def Hybrid_Rag(query: str):

    """
    Retrieve relevant chunks using
    Chroma + BM25.
    """

    initialize_rag()

    llm = get_llm()

    prompt = f"""
    Refine the user query.

    Query:
    {query}
    """

    query_refine = llm.invoke(
        prompt
    ).content

    result = collection.query(
        query_texts=[query_refine],
        n_results=3
    )

    distances = result["distances"][0]
    documents = result["documents"][0]

    threshold = 1.0

    near_chunks = []

    for dist, doc in zip(
        distances,
        documents
    ):

        if dist < threshold:
            near_chunks.append(doc)

    scores = token_cor.get_scores(
        query_refine.split()
    )

    indexed = list(
        enumerate(scores)
    )

    ranked = sorted(
        indexed,
        key=lambda x: x[1],
        reverse=True
    )

    top_idx = [
        idx
        for idx, _
        in ranked[:10]
    ]

    bm25_docs = [
        chunks[i]
        for i in top_idx
    ]

    rrf = {}

    for rank, doc in enumerate(
        near_chunks
    ):
        rrf[doc] = (
            rrf.get(doc, 0)
            + 1 / (rank + 60)
        )

    for rank, doc in enumerate(
        bm25_docs
    ):
        rrf[doc] = (
            rrf.get(doc, 0)
            + 1 / (rank + 60)
        )

    merged = sorted(
        rrf.items(),
        key=lambda x: x[1],
        reverse=True
    )

    hybrid_docs = [
        doc
        for doc, _
        in merged[:5]
    ]

    if hybrid_docs:
        return "\n\n".join(
            hybrid_docs
        )

    from tavily import TavilyClient

    tavily = TavilyClient(
        api_key=os.getenv(
            "TAVILY_API_KEY"
        )
    )

    response = tavily.search(
        query=query_refine
    )

    return str(
        response["results"]
    )


# =========================
# Research Tool
# =========================

@mcp.tool()
def Research_tool(query: str):

    docs = Hybrid_Rag(query)

    llm = get_llm()

    prompt = f"""
    Answer the question using
    the context below.

    Context:
    {docs}

    Question:
    {query}
    """

    return llm.invoke(
        prompt
    ).content


# =========================
# MCP Server
# =========================

if __name__ == "__main__":
    mcp.run()