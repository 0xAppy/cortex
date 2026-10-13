from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

tavily = TavilySearch(max_results=3)

response = tavily.invoke({"query": "What is RAG in AI?"})

print(response)