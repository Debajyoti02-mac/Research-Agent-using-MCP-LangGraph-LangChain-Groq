from langchain_mcp_adapters.client import MultiServerMCPClient 
import warnings
warnings.filterwarnings('ignore') 

from langgraph.prebuilt import create_react_agent

import os 
from dotenv import load_dotenv 
from langchain_groq import ChatGroq 
load_dotenv()
from langgraph.checkpoint import memory 
memory = memory.MemorySaver()
import asyncio 
async def main ():
    client = MultiServerMCPClient({
        'Research_Agent':{
        'command':'python',
        'args':['Agent_Server.py'],
        'transport':'stdio'
    }
    })

    LLM = ChatGroq(model="openai/gpt-oss-120b") 
    tool  = await client.get_tools() 
    if not tool:
        print("tools cant fetch by server now")
    system_prompt = "YOU are a realiable ai assistent for This MCP server so do the work sufficiently one by one using tool"
    research_agent_create=create_react_agent(model=LLM , tools=tool , prompt=system_prompt,checkpointer=memory)

    config = {
        "configurable":{
            "thread_id":"research_002"
        }
    }

# user query 
    query = " what is RAG full form ? "
    print(f'user given query is : {query}') 

    response = await research_agent_create.ainvoke({"messages":[("user",query)]},config=config)
    print(response['messages'][-1].content)
# call the server
if __name__=="__main__":
    asyncio.run(main())