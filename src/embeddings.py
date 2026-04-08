from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Loads and returns the HuggingFace embedding model.
    This single instance is reused across:
    - ChromaDB for storing vectors
    - EmbeddingsFilter for compression
    """
    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},

    )
    print("Embedding model loaded successfully.")
    return embeddings
