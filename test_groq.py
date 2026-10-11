from dotenv import load_dotenv
from langchain_groq import ChatGroq

## Load Environment Variables
load_dotenv()

## Initialize Model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

## Test Groq
response = llm.invoke("In one sentence, what is a RAG-based AI assistant?")
print(response.content)
