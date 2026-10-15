from sentence_transformers import SentenceTransformer
from fastmcp import FastMCP
import chromadb


## Models & Database
model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="data/chroma")
collection = chroma_client.get_collection(name='cortex_kb')

## MCP Server
mcp = FastMCP("Cortex Retriever")

## Retriever Tool
@mcp.tool()
def retrieve_documents(question: str) -> list[dict]:
    """searches Cortex's local knowledge base and returns the most relevant chunks for a question"""
    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings= [query_embedding],
        n_results= 3
    )
    
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    
    retrieved_chunks = []
    
    for doc_id, document, metadata in zip(ids, documents, metadatas):
        retrieved_chunks.append({
        "id": doc_id,
        "text": document,
        "source": metadata["source"],
        "chunk_index": metadata["chunk_index"]
        })
    
    return retrieved_chunks
    
if __name__ == "__main__":
    mcp.run()