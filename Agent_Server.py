from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Research_Agent")
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
        raise TypeError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            return _OPERATORS[op_type](operand)
        raise TypeError(f"Unsupported unary operator: {op_type.__name__}")
    elif isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise NameError(f"Unknown constant: {node.id}")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
            args = [_safe_eval_node(arg) for arg in node.args]
            return _FUNCTIONS[node.func.id](*args)
        raise NameError(f"Unsupported function: {node.func}")
    else:
        raise TypeError(f"Unsupported expression element: {type(node).__name__}")

# Calculator Tool 
@mcp.tool() 
def calculator(exec: str):
    """Safely evaluates mathematical and arithmetic expressions. Supports +, -, *, /, %, **, ^, parentheses, sqrt, sin, cos, log, pi, e."""
    try:
        clean_expr = exec.strip().replace("^", "**")
        tree = ast.parse(clean_expr, mode="eval")
        result = _safe_eval_node(tree)
        return str(result)
    except Exception as e: 
        return f"Calculation error: {e}" 


# Weather tool
import requests  
@mcp.tool()
def weather(location :str):
    """ Provided users location's weather fetched by this tool """
    try : 
        response = requests.get(f"https://wttr.in/{location}?format=3").text
        return response
    except Exception as e :
        return str(e)

# File I/O Operations 
import os
@mcp.tool()
def file_read(filename: str):
    """Reads the contents of a local file from the file system. Example: 'notes.txt'"""
    try:
        with open(file=filename, mode="r", encoding="utf-8", errors="ignore") as f:
            return f"Content of {filename}:\n{f.read()}"
    except Exception as e:
        return f"Error reading file '{filename}': {e}"

@mcp.tool()
def write_file(filename: str, content: str):
    """Creates or overwrites a local file on the file system with the given content."""
    try:
        dir_name = os.path.dirname(os.path.abspath(filename))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(file=filename, mode="w", encoding="utf-8") as file:
            bytes_written = file.write(content)

        return f"Successfully wrote {bytes_written} characters to '{filename}'."
    except Exception as e:
        return f"Error writing to file '{filename}': {e}"


# Load document 
from langchain_community.document_loaders import PyPDFLoader 
loader = PyPDFLoader('2005.11401v4.pdf')
pages = loader.load()
# split document  
from langchain_text_splitters import RecursiveCharacterTextSplitter 
spliter = RecursiveCharacterTextSplitter(chunk_overlap=180 , chunk_size=1200)
texts = spliter.split_documents(pages)
chunks = [i.page_content for i in texts]
metadata = [i.metadata for i in texts]

# Use Chromadb 
import chromadb 
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction 
embedding_fun = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path='./Research_Agent')

import warnings 
warnings.filterwarnings('ignore')
try:
    collection = client.get_or_create_collection(name="Agent",embedding_function=embedding_fun)
    if collection.count()==0:
        collection.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks , 
            metadatas=[i.metadata for i in texts]
        )
    print(f"sucessfullt created the collection is : {collection.count()}")
except Exception as e :
    print(str(e))

# For Hybrid RAG 
from rank_bm25 import BM25Okapi 
token = [i.split() for i in chunks]
token_cor = BM25Okapi(token)

# import LLM 
import os 
from dotenv import load_dotenv 
from langchain_groq import ChatGroq 
load_dotenv()
key=os.getenv("GROQ_API_KEY")
LLM = ChatGroq(model="openai/gpt-oss-120b")

# Retrival Tool 
import os 
from dotenv import load_dotenv
load_dotenv()
@mcp.tool()
def Hybrid_Rag(query:str):
    """ 
    Before Giving out any answers follow this steps : 
    user's asking questions always follow through these steps
    step 1 :--->  first enter in Hybrid_Rag 
    step 2 :---> search for nearest chunking 
    step 3 :--> got then return the chunks to the user 
    step 4 :---> either goes to the websearch tool [if cant find any related content]
    """
    prompt = f""" 
    refine the user asking and make it context understable : 
    query : {query}
    """
    query_refine = LLM.invoke(prompt).content

    result = collection.query(query_texts=[query_refine],n_results=3)
    distance = result['distances'][0]
    document = result['documents'][0]
    threshold = 1.0
    near_chunks = []

    for dist, doc in zip(distance, document):
        if dist < threshold:
            near_chunks.append(doc)

    score = token_cor.get_scores(query_refine.split())
    def get_scores(score , k=10):
        index  = list(enumerate(score))
        sorted_index = sorted(index , key = lambda x:x[1], reverse=True)
        return [doc for doc , i in sorted_index[:k]]

    get_top_scores = get_scores(score , k=10)
    copy_top_scores = [chunks[i] for i in get_top_scores]

    rrf_token={}

    for rank , doc in enumerate(near_chunks):
        rrf_token[doc]=rrf_token.get(doc,0)+1/(rank+60)

    for rank , doc in enumerate(copy_top_scores):
        rrf_token[doc]=rrf_token.get(doc,0)+1/(rank+60)
    marge = sorted(rrf_token.items() , key=lambda x:x[1] , reverse=True)
    hybrid_top_docs=[doc for doc, _ in marge[:5]]
    if hybrid_top_docs:
        return "\n\n".join(hybrid_top_docs)

    from tavily import TavilyClient

    client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )
    response = client.search(query=query_refine)
    return str(response['results'])

@mcp.tool()
def Research_tool(query: str):
    docs = Hybrid_Rag(query)

    prompt = f"""
    Answer only the user's question.

    Context:
    {docs}

    Question:
    {query}
    """

    return LLM.invoke(prompt).content

# Runing MCP server
if __name__ == "__main__":
    mcp.run()