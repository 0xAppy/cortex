import asyncio
from pydantic import BaseModel
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import json

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
    
## LLM Setup
load_dotenv()

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
    print("Knowledge insufficient — routed to approval.")
    return {"approved": False}

## Graph Workflow 
builder.add_node("retrieve", retrieve_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("generate", generate_node)
builder.add_node("approval", approval_node)


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
builder.add_edge("approval", END)
builder.add_edge("generate", END)
graph = builder.compile()

## Run Graph
async def main():
    result = await graph.ainvoke({
        "question": "Who won the FIFA World Cup in 1962?",
        "kb_results": [],
        "answer": "",
        "needs_web_search": False,
        "approved": False
    })

    print("Sufficiency:", not result["needs_web_search"])
    print("Answer:", result["answer"])


if __name__ == "__main__":
    asyncio.run(main())
