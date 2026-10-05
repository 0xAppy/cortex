# Cortex

Cortex is a RAG-based AI assistant designed to answer questions using a local knowledge base.

The system stores documents in a vector database and uses embeddings to represent text as numerical vectors. These vectors allow Cortex to retrieve information based on semantic similarity rather than only matching exact words.

Cortex uses ChromaDB as the vector database. Documents are divided into smaller chunks before they are embedded and stored. When a user asks a question, the question is also converted into an embedding. Cortex then searches ChromaDB for chunks that are semantically similar to the question.

The retrieved chunks will eventually be provided to a language model as context so that the model can generate an answer grounded in the user's knowledge base.