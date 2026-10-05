# Embeddings

An embedding is a numerical representation of text that captures semantic information.

Cortex uses the all-MiniLM-L6-v2 embedding model. This model converts text into a 384-dimensional vector.

Texts with similar meanings tend to have vectors that are closer together in the embedding space. This allows a retrieval system to find relevant information even when the user's question does not contain the exact words used in the original document.

For example, a question about forgetting login credentials can retrieve a document discussing password recovery even if the two texts use different words.