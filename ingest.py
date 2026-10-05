# =========================
# Imports
# =========================

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


# =========================
# Configuration
# =========================

RAW_DIR = Path("data/raw")


# =========================
# Models & Database
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="data/chroma")
collection = chroma_client.get_or_create_collection(name="cortex_kb")


# =========================
# Find Documents
# =========================

files = list(RAW_DIR.glob("*.md"))


# =========================
# Chunking
# =========================

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start = end - overlap

    return chunks


# =========================
# Process Documents
# =========================

for file in files:
    text = file.read_text(encoding="utf-8")
    chunks = chunk_text(text)

    print(f"\n--- {file.name} ---")
    print(f"Number of chunks: {len(chunks)}")

    embeddings = model.encode(chunks)
    
    ids = [f"{file.stem}_{i}" for i in range(len(chunks))]

    collection.add(
        ids = ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimensions: {len(embeddings[0])}")
    print(f"First embedding: {embeddings[0][:10]}")

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(chunk)

# =========================
# Verify Database
# =========================

print(f"\nTotal chunks in ChromaDB: {collection.count()}")

## Query Knowledge Base
question = input("\nAsk Cortex a question: ")

query_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings = [query_embedding],
    n_results=3
)

print("\nRelevant chunks:")

for i, document in enumerate(results["documents"][0]):
    print(f"\nResult {i + 1}:")
    print(document)