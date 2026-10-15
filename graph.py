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
from pydantic import BaseModel, Field
from langgraph.types import interrupt
from langgraph.types import Command

## LLM Setup
load_dotenv()

## Agent State
class AgentState(TypedDict):
    question: str
    kb_results: list[dict]
    answer: str
    needs_web_search: bool
    approved: bool
    web_results: list


## Sufficiency Decision
class SufficiencyDecision(BaseModel):
    sufficient: bool
    

## Structured Response Schema
class CortexResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

    
## Web Search Tool
tavily = TavilySearch(max_results=3)

## Web Search Node
def web_search_tool(state: AgentState) -> dict:
    question = state["question"]
    response = tavily.invoke({"query": question})
    print("Tavily Search Completed!")
    return {"web_results" : response}


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

evaluator = llm.with_structured_output(SufficiencyDecision)

## Rejection Node
def rejection_node(state: AgentState) -> dict:
    return {
        "answer": (
            "Web search was not approved. "
            "I couldn't find enough information in the local knowledge base "
            "to answer your question reliably."
        )
    }
    
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

## Router After Evaluation
def route_after_evaluation(state: AgentState) -> str:
    if state["needs_web_search"]:
        return "approval"

    return "generate"

## Router After Approval
def route_after_approval(state: AgentState) -> str:
    if state["approved"]:
        return "web_search"

    return "rejected"
    
## Generate Web Answer Node
def generate_web_answer_node(state: AgentState) -> dict:
    question = state["question"]
    web_results = state["web_results"]

    response = llm.invoke([
        (
            "system",
            "Answer the question using the provided web search results. "
            "Be accurate and do not invent facts. If the results are "
            "insufficient, say so. Include relevant source URLs when available."
        ),
        (
            "human",
            f"Question: {question}\n\nWeb search results:\n{web_results}"
        )
    ])

    return {"answer": response.content}

## Graph Builder
builder = StateGraph(AgentState)

## Approval Node
def approval_node(state: AgentState) -> dict:
    decision = interrupt(
        "Knowledge is insufficient. Approve web search?"
    )
    
    return {"approved": decision}

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

## Graph Workflow 
builder.add_node("retrieve", retrieve_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("generate", generate_node)
builder.add_node("approval", approval_node)
builder.add_node("web_search", web_search_tool)
builder.add_node("rejected", rejection_node)
builder.add_node("generate_web_answer", generate_web_answer_node)


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
## Route After Approval
builder.add_conditional_edges(
    "approval",
    route_after_approval,
    {
        "web_search": "web_search",
        "rejected": "rejected"
    }
)
builder.add_edge("web_search", "generate_web_answer")
builder.add_edge("generate_web_answer", END)
builder.add_edge("rejected", END)
builder.add_edge("generate", END)

## Checkpointer
memory = InMemorySaver()

## Compile Graph
graph = builder.compile(checkpointer=memory)

## Run Graph

## Run Graph
async def main():
    config = {"configurable": {"thread_id": "cortex-test-1"}}

    result = await graph.ainvoke(
        {
            "question": "Why does Cortex use ChromaDB?",
            "kb_results": [],
            "answer": "",
            "needs_web_search": False,
            "approved": False,
            "web_results": []
        },
        config=config
    )

    ## Human Approval
    snapshot = graph.get_state(config)

    if snapshot.next:
        decision = input("Allow web search? (yes/no): ").strip().lower()

        if decision == "yes":
            result = await graph.ainvoke(
                Command(resume=True),
                config=config
            )
        else:
            result = await graph.ainvoke(
                Command(resume=False),
                config=config
            )

    print("Sufficiency:", not result["needs_web_search"])
    print("Approved:", result["approved"])
    print("Answer:", result["answer"])

    
if __name__ == "__main__":
    asyncio.run(main())
    

## Temporary Structured Output Test
test_llm = llm.with_structured_output(CortexResponse)

test_response = test_llm.invoke(
    "Explain in one sentence why Cortex uses ChromaDB. "
    "Return one citation named 'cortex_overview.md' "
    "and a confidence score of 0.95."
)

print(test_response)
print(type(test_response))

