from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class AgentState(TypedDict):
    question : str
    kb_result : list[str]
    answer : str
    needs_web_search : bool
    approved : bool
     