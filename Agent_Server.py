try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("Research_Agent")
except Exception:
    class DummyMCP:
        def tool(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator

        def run(self, *args, **kwargs):
            pass

    mcp = DummyMCP()

import ast
import operator
import math
import os
import re
import requests

from dotenv import load_dotenv

load_dotenv()

# =====================================
# Calculator
# =====================================

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
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "abs": abs,
    "round": round,
    "pow": math.pow,
    "factorial": math.factorial,
    "comb": math.comb,
    "perm": math.perm,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "degrees": math.degrees,
    "radians": math.radians,
    "hypot": math.hypot,
    "max": max,
    "min": min,
    "sum": sum,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
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

    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            return _OPERATORS[op_type](operand)

    elif isinstance(node, ast.Name):
        name_lower = node.id.lower()
        if name_lower in _CONSTANTS:
            return _CONSTANTS[name_lower]
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]

    elif isinstance(node, ast.Attribute):
        # Support math.pi, math.sqrt, etc.
        attr_lower = node.attr.lower()
        if attr_lower in _CONSTANTS:
            return _CONSTANTS[attr_lower]
        if attr_lower in _FUNCTIONS:
            return _FUNCTIONS[attr_lower]

    elif isinstance(node, ast.Call):
        func = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
            if func_name in _FUNCTIONS:
                func = _FUNCTIONS[func_name]
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr.lower()
            if func_name in _FUNCTIONS:
                func = _FUNCTIONS[func_name]

        if func is not None:
            args = [_safe_eval_node(arg) for arg in node.args]
            return func(*args)

    raise TypeError(f"Unsupported expression component: {type(node).__name__}")


@mcp.tool()
def calculator(expression: str = "", exec: str = "", expr: str = "", **kwargs):
    """Evaluate mathematical expressions safely. Accepts expressions like '256 * 48 + 1024', 'sqrt(144) + pi', 'factorial(5)'."""
    try:
        raw = expression or exec or expr or (list(kwargs.values())[0] if kwargs else "")
        if not raw:
            return "Error: No mathematical expression provided."

        cleaned = str(raw).strip()
        # Handle multiplication/division signs and power operators
        cleaned = cleaned.replace("×", "*").replace("÷", "/").replace("^", "**")
        # Replace 'x' or 'X' when used as multiplication operator between numbers (e.g. 25 x 4 -> 25 * 4)
        cleaned = re.sub(r'(?<=\d)\s*[xX]\s*(?=\d)', ' * ', cleaned)
        # Remove digit grouping commas (e.g. 1,000 -> 1000)
        cleaned = re.sub(r'(?<=\d),(?=\d)', '', cleaned)

        tree = ast.parse(cleaned, mode="eval")
        res = _safe_eval_node(tree)

        if isinstance(res, float) and res.is_integer():
            return str(int(res))
        elif isinstance(res, float):
            return f"{res:.10g}"
        return str(res)

    except Exception as e:
        return f"Calculation error: {str(e)}"


# =====================================
# Weather
# =====================================

@mcp.tool()
def weather(location: str):
    """Get weather information."""

    try:
        return requests.get(
            f"https://wttr.in/{location}?format=3"
        ).text

    except Exception as e:
        return str(e)


# =====================================
# File Read
# =====================================

@mcp.tool()
def file_read(filename: str):
    """Read content from a local file. Example: 'notes.txt', 'root.txt'."""
    try:
        clean_name = str(filename).strip().strip("\"'")
        if not os.path.exists(clean_name):
            return f"Error: File '{clean_name}' not found on filesystem."

        with open(clean_name, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        return content

    except Exception as e:
        return f"Error reading file '{filename}': {str(e)}"


@mcp.tool()
def read_file(filename: str):
    """Read content from a local file. Example: 'notes.txt', 'root.txt'."""
    return file_read(filename)


# =====================================
# File Write
# =====================================

@mcp.tool()
def write_file(filename: str, content: str):
    """Create or update a local file with the provided text content. Example: 'notes.txt'."""
    try:
        clean_name = str(filename).strip().strip("\"'")

        if len(content) > 100000:
            return f"Error: Content too large ({len(content)} chars, max allowed is 100,000 chars)."

        parent_dir = os.path.dirname(clean_name)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        with open(clean_name, "w", encoding="utf-8") as f:
            f.write(content)

        lines_count = len(content.splitlines())
        return f"File '{clean_name}' saved successfully ({len(content)} characters, {lines_count} lines)."

    except Exception as e:
        return f"Error writing to file '{filename}': {str(e)}"


@mcp.tool()
def file_write(filename: str, content: str):
    """Create or update a local file with the provided text content. Example: 'notes.txt'."""
    return write_file(filename, content)


# =====================================
# Lazy Globals
# =====================================

collection = None
LLM = None


# =====================================
# Lazy LLM
# =====================================

def get_llm(groq_api_key: str = None):
    global LLM

    key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set. Please set the GROQ_API_KEY environment variable or enter it in the app.")

    if LLM is None or groq_api_key:
        from langchain_groq import ChatGroq

        LLM = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            groq_api_key=key,
        )

    return LLM


# =====================================
# Load Existing Chroma Only
# =====================================

def initialize_rag():
    global collection

    if collection is not None:
        return

    import chromadb

    cohere_key = os.getenv("COHERE_API_KEY")
    if cohere_key:
        try:
            from chromadb.utils.embedding_functions import CohereEmbeddingFunction
            embedding_function = CohereEmbeddingFunction(
                api_key=cohere_key,
                model_name="embed-english-v3.0",
            )
        except Exception:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            embedding_function = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
    else:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    client = chromadb.PersistentClient(path="./Research_Agent")

    try:
        collection = client.get_collection(
            name="Agent",
            embedding_function=embedding_function,
        )
    except Exception:
        collection = client.get_or_create_collection(
            name="Agent",
            embedding_function=embedding_function,
        )

    print("Existing Chroma loaded.")


# =====================================
# Hybrid RAG
# =====================================

@mcp.tool()
def Hybrid_Rag(query: str):

    """
    Retrieve relevant information
    from existing ChromaDB.
    """

    initialize_rag()

    llm = get_llm()

    query_refine = llm.invoke(
        f"Refine this query:\n{query}"
    ).content

    result = collection.query(
        query_texts=[query_refine],
        n_results=5
    )

    docs = result["documents"][0]

    if docs:
        return "\n\n".join(docs)

    from tavily import TavilyClient

    tavily = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    response = tavily.search(
        query=query_refine
    )

    return str(response["results"])


# =====================================
# Research Tool
# =====================================

@mcp.tool()
def Research_tool(query: str):

    context = Hybrid_Rag(query)

    llm = get_llm()

    prompt = f"""
Answer the question using the context.

Context:
{context}

Question:
{query}
"""

    return llm.invoke(prompt).content


# =====================================
# Run MCP
# =====================================

if __name__ == "__main__":
    mcp.run()