import asyncio
import json
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

## LLM Setup
load_dotenv()

## Agent State
class AgentState(TypedDict):
    question: str
    kb_results: list[str]
    answer: str
    needs_web_search: bool
    approved: bool


## Sufficiency Decision
class SufficiencyDecision(BaseModel):
    sufficient: bool
    
## Web Search Tool
tavily = TavilySearch(max_results=3)

## Web Search Node
def web_search_tool(state: AgentState) -> dict:
    question = state["question"]
    response = tavily.invoke({"query": question})
    return {"web_results" : response}


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

evaluator = llm.with_structured_output(SufficiencyDecision)
    
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
        
## Evaluate Node
def evaluate_node(state: AgentState) -> dict:
    question = state["question"]
    kb_results = state["kb_results"]

    response = evaluator.invoke([
        (
            "system",
            "Decide whether the provided knowledge chunks contain enough "
            "information to answer the question. Do not use outside knowledge. "
            "Return sufficient=True only when the chunks support an answer."
        ),
        (
            "human",
            f"Question: {question}\n\nKnowledge chunks:\n{kb_results}"
        )
    ])

    return {"needs_web_search": not response.sufficient}

## Router
def route_after_evaluation(state: AgentState) -> str:
    if state["needs_web_search"]:
        return "approval"
    return "generate"
    
## Generate Node
def generate_node(state: AgentState) -> dict:
    question = state["question"]
    kb_results = state["kb_results"]

    response = llm.invoke([
        (
            "system",
            "Answer the question using only the provided knowledge chunks. "
            "If the chunks do not contain the answer, say that the information "
            "is not available in the provided context. Do not invent facts."
        ),
        (
            "human",
            f"Question: {question}\n\nKnowledge chunks:\n{kb_results}"
        )
    ])

    return {"answer": response.content}

## Graph Builder
builder = StateGraph(AgentState)

## Approval Node
def approval_node(state: AgentState) -> dict:
    print("Knowledge insufficient — web search approval required")
    return {}

## Graph Workflow 
builder.add_node("retrieve", retrieve_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("generate", generate_node)
builder.add_node("approval", approval_node)
builder.add_node("web_search", web_search_tool)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "evaluate")

builder.add_conditional_edges(
    "evaluate",
    route_after_evaluation,
    {
        "generate": "generate",
        "approval": "approval"
    }
)
builder.add_edge("approval", "web_search")
builder.add_edge("web_search", END)
builder.add_edge("generate", END)

## Checkpointer
memory = InMemorySaver()

## Compile Graph
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["web_search"]
    )

## Run Graph
async def main():
    config = {"configurable": {"thread_id": "cortex-test-1"}}
    result = await graph.ainvoke({
        "question": "Who won the FIFA World Cup in 1962?",
        "kb_results": [],
        "answer": "",
        "needs_web_search": False,
        "approved": False,
        "web_results": []
    },
        config = config                        
    )
    ## Check Graph State
    snapshot = graph.get_state(config)

    print("Next node:", snapshot.next)
    print("Current state:", snapshot.values)


if __name__ == "__main__":
    asyncio.run(main())
