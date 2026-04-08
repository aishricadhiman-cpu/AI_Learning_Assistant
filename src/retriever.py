from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    MMR_K,
    MMR_FETCH_K,
    MMR_LAMBDA,
    SIMILARITY_THRESHOLD
)

def get_retriever(
        vectorstore : Chroma,
        embeddings: HuggingFaceEmbeddings
) -> ContextualCompressionRetriever:
    """
        Builds two stage retriever:
        Stage 1 — MMR retriever for diverse relevant chunks
        Stage 2 — EmbeddingsFilter to remove noise from chunks
    """

    mmr_retriever = vectorstore.as_retriever(
        search_type = 'mmr',
        search_kwargs ={
            "k": MMR_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA
        }
    )

    embeddings_filter = EmbeddingsFilter(
        embeddings = embeddings,
        similarity_threshold = SIMILARITY_THRESHOLD
    )

    final_retriever = ContextualCompressionRetriever(
        base_compressor = embeddings_filter,
        base_retriever = mmr_retriever
    )
    print("retriever ready")
    return final_retriever

