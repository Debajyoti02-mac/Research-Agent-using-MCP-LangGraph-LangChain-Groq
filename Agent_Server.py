from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Research_Agent")
import ast

# Calculator Tool 
@mcp.tool() 
def calculator(exec:str):
    """ user given all aithmetic operations performed by this tool """
    try : 
        return str(ast.literal_eval(exec))
    except Exception as e : 
        return str(e) 


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
def file_read(filename:str):
    """ file read by this tool """
    with open (file=filename) as f :
        return f" content is : {f.read()}"

@mcp.tool()
def write_file(filename:str,content:str):
    """ file writed by this tool . 
    if file isn't existing then create it and also write the user given content on that
    """
    os.makedirs(
        os.path.dirname(os.path.abspath(filename)),
        exist_ok=True
    )

    with open(file=filename, mode="w") as file:
        res = file.write(content)

    return f"sucessfully created : {res}"


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