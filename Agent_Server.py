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


_SAFE_MATH_GLOBALS = {
    "__builtins__": None,
    "abs": abs,
    "round": round,
    "max": max,
    "min": min,
    "sum": sum,
    "math": math,
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
    "pow": math.pow,
    "factorial": math.factorial,
    "comb": math.comb,
    "perm": math.perm,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "degrees": math.degrees,
    "radians": math.radians,
    "hypot": math.hypot,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


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

        # Primary: AST safe evaluation
        try:
            tree = ast.parse(cleaned, mode="eval")
            res = _safe_eval_node(tree)
        except Exception:
            # Fallback: Restricted safe eval without builtins
            res = eval(cleaned, _SAFE_MATH_GLOBALS, {})

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
# =====================================
# File Read
# =====================================

def _find_file_case_insensitive(filename: str, search_dirs=("uploads", ".")):
    target = os.path.basename(filename).strip().strip("\"'").lower()
    for d in search_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower() == target:
                    return os.path.join(d, f)
                if os.path.splitext(f)[0].lower() == os.path.splitext(target)[0].lower():
                    return os.path.join(d, f)
    # If single document exists in uploads
    if os.path.exists("uploads") and os.path.isdir("uploads"):
        files = [f for f in os.listdir("uploads") if not f.startswith(".")]
        if len(files) == 1:
            return os.path.join("uploads", files[0])
    return None


@mcp.tool()
def file_read(filename: str = "", file_path: str = "", path: str = "", **kwargs):
    """Read content from a local file or PDF document. Example: 'notes.txt', 'cv.pdf', 'paper.pdf'."""
    try:
        raw_name = filename or file_path or path or kwargs.get("name") or kwargs.get("filepath") or (list(kwargs.values())[0] if kwargs else "")
        clean_name = str(raw_name).strip().strip("\"'")

        if not clean_name:
            # Check if there is a file in uploads directory to default to
            if os.path.exists("uploads") and os.path.isdir("uploads"):
                files = [f for f in os.listdir("uploads") if not f.startswith(".")]
                if files:
                    clean_name = files[0]
            if not clean_name:
                return "Error: No filename provided to read."

        found_path = None
        if os.path.exists(clean_name) and os.path.isfile(clean_name):
            found_path = clean_name
        else:
            found_path = _find_file_case_insensitive(clean_name)

        if found_path:
            if found_path.lower().endswith(".pdf"):
                import pypdf
                reader = pypdf.PdfReader(found_path)
                text = ""
                for idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    text += f"\n--- Page {idx + 1} ---\n" + page_text
                return text.strip() if text.strip() else "PDF loaded, but no readable text found."
            else:
                with open(found_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()

        # If not found on local disk, search ChromaDB vector store
        initialize_rag()
        if collection is not None and collection.count() > 0:
            try:
                query_res = collection.get(where={"source": os.path.basename(clean_name)})
                docs = query_res.get("documents", [])
                if docs:
                    return "\n\n---\n\n".join(docs)
            except Exception:
                pass

            rag_res = Hybrid_Rag(clean_name)
            if rag_res and "No matching" not in rag_res and "No indexed" not in rag_res:
                return rag_res

        return f"Error: File '{clean_name}' not found on filesystem or knowledge base."

    except Exception as e:
        return f"Error reading file '{filename}': {str(e)}"


@mcp.tool()
def read_file(filename: str = "", file_path: str = "", path: str = "", **kwargs):
    """Read content from a local file. Example: 'notes.txt', 'root.txt'."""
    return file_read(filename=filename, file_path=file_path, path=path, **kwargs)


# =====================================
# File Write
# =====================================

@mcp.tool()
def write_file(filename: str = "", content: str = "", file_path: str = "", text: str = "", **kwargs):
    """Create or update a file with the provided text content. Saves in 'uploads/' directory by default. Example: 'notes.txt'."""
    try:
        fname = filename or file_path or kwargs.get("path") or kwargs.get("name") or "output.txt"
        cnt = content or text or kwargs.get("data") or kwargs.get("body") or ""
        clean_name = str(fname).strip().strip("\"'")

        if len(cnt) > 100000:
            return f"Error: Content too large ({len(cnt)} chars, max allowed is 100,000 chars)."

        # If a direct filename without directory is given, save in uploads/
        if not os.path.dirname(clean_name):
            os.makedirs("uploads", exist_ok=True)
            target_path = os.path.join("uploads", clean_name)
        else:
            target_path = clean_name
            parent_dir = os.path.dirname(target_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(cnt)

        lines_count = len(cnt.splitlines())
        return f"File '{clean_name}' saved successfully in '{target_path}' ({len(cnt)} characters, {lines_count} lines)."

    except Exception as e:
        return f"Error writing to file '{filename}': {str(e)}"


@mcp.tool()
def file_write(filename: str = "", content: str = "", file_path: str = "", text: str = "", **kwargs):
    """Create or update a file with the provided text content. Saves in 'uploads/' by default. Example: 'notes.txt'."""
    return write_file(filename=filename, content=content, file_path=file_path, text=text, **kwargs)


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

    client = chromadb.PersistentClient(path="./Research_Agent")

    # 1. Try loading existing collection using its persisted configuration
    try:
        collection = client.get_collection(name="Agent")
        print(f"Existing Chroma collection 'Agent' loaded ({collection.count()} chunks).")
        return
    except Exception:
        pass

    # 2. If collection doesn't exist yet, determine embedding function and create it
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

    collection = client.get_or_create_collection(
        name="Agent",
        embedding_function=embedding_function,
    )
    print(f"Chroma collection 'Agent' initialized ({collection.count()} chunks).")


# =====================================
# Hybrid RAG
# =====================================

@mcp.tool()
def Hybrid_Rag(query: str):
    """
    Retrieve relevant excerpts and research context from indexed documents, papers, or uploaded PDFs in ChromaDB.
    """
    initialize_rag()

    if collection is None or collection.count() == 0:
        return "No indexed documents found in the database. Please upload a PDF or document in the sidebar first."

    n_results = min(5, max(1, collection.count()))
    try:
        result = collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        docs = result.get("documents", [[]])[0]
        if docs:
            return "\n\n---\n\n".join(docs)
    except Exception as e:
        print(f"ChromaDB query warning: {e}")

    # Fallback to Tavily web search if API key is provided
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from tavily import TavilyClient
            tavily = TavilyClient(api_key=tavily_key)
            response = tavily.search(query=query)
            results = response.get("results", [])
            if results:
                return "\n\n".join([f"**{r.get('title', '')}**\n{r.get('content', '')}" for r in results])
        except Exception:
            pass

    return "No matching document excerpts found in the knowledge base."


# =====================================
# Research Tool
# =====================================

@mcp.tool()
def Research_tool(query: str):
    """
    Answer research and document questions using context retrieved from the indexed knowledge base.
    """
    context = Hybrid_Rag(query)

    llm = get_llm()

    prompt = f"""Use the following retrieved context from the uploaded documents to answer the question clearly, concisely, and accurately.

Retrieved Context:
{context}

Question:
{query}

Answer:"""

    return llm.invoke(prompt).content


# =====================================
# Run MCP
# =====================================

if __name__ == "__main__":
    mcp.run()