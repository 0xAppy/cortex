import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import asyncio
import json

## Agent State
class AgentState(TypedDict):
    question : str
    kb_results : list[str]
    answer : str
    needs_web_search : bool
    approved : bool

## Retrieve Node
async def retrieve_node(state: AgentState) -> dict:     
    # Configure the MCP server
    cortex = StdioServerParameters(
            command="python",
            args=["mcp_server.py"]
    )

    # Extract the question from graph state
    question = state["question"]
    
    # Connect to the MCP server
    async with stdio_client(cortex) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            
            # Initialize the MCP session
            await session.initialize()
            
            # Call the retriever tool
            response_tool = await session.call_tool(
                name = "retrieve_documents",
                arguments={"question": question}
            )

            # Extract the returned text
            result = response_tool.structured_content["result"]
            return {"kb_results": result}
            
## Graph Builder
builder = StateGraph(AgentState)

## Graph Workflow 
builder.add_node("retrieve", retrieve_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", END)

graph = builder.compile()
            

## Run Graph
async def main():
    result = await graph.ainvoke({
        "question": "Why does Cortex use ChromaDB?",
        "kb_results": [],
        "answer": "",
        "needs_web_search": False,
        "approved": False
    })

    print(result["kb_results"])


if __name__ == "__main__":
    asyncio.run(main())
