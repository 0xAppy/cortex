# ChromaDB

ChromaDB is a vector database that can store embeddings and the text associated with those embeddings.

Cortex uses a persistent ChromaDB client so that the stored knowledge remains available between program runs. This means the application does not need to recreate and re-embed every document whenever it starts.

A ChromaDB collection acts as a container for related documents and their embeddings. Documents can be added to a collection and later queried using an embedding-based similarity search.

The query process compares the embedding of a user's question with stored embeddings and returns the most relevant document chunks.