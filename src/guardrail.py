import logging
from config import LOG_FILE
import os
from src.cleaner import preprocess_query

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename = LOG_FILE,
    level = logging.INFO,
    format = "%(asctime)s — %(message)s"
)

ML_KEYWORDS = [
    "machine learning", "deep learning", "neural network",
    "supervised", "unsupervised", "reinforcement",
    "regression", "classification", "clustering",
    "cnn", "rnn", "lstm", "transformer", "autoencoder", "gan",
    "training", "overfitting", "underfitting", "regularization",
    "dropout", "batch normalization", "epoch", "batch",
    "optimizer", "gradient", "backpropagation", "loss", "activation",
    "keras", "tensorflow", "sklearn", "scikit",
    "accuracy", "precision", "recall", "f1", "roc", "auc",
    "confusion matrix", "cross validation",
    "dataset", "feature", "label", "preprocessing", "normalization",
    "perceptron", "dense", "convolutional", "pooling",
    "embedding", "attention", "encoder", "decoder",
    "model", "predict", "fit", "weights", "bias",
    "learning rate", "momentum", "Adam", "SGD",
    "softmax", "sigmoid", "relu", "tanh"
]

def is_relevant_query(query: str) -> bool:
    """
    Returns True if the query contains ML/DL keywords
    Returns False if completely off-topic

    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ML_KEYWORDS)

def get_answer_with_sources(chain,question: str) -> str:
    """
    Runs chain, handles fallback if no chunks found
    Returns answer with source citations

    """

    cleaned_question = preprocess_query(question)

    logging.info(f"ORIGINAL QUERY: {question}")
    logging.info(f"CLEANED QUERY: {cleaned_question}")

    response = chain.invoke({"question": question})

    answer = response["answer"]
    source_docs = response["source_documents"]

    logging.info(f"CHUNKS_RETRIEVED: {len(source_docs)}")

    if not source_docs:
        logging.warning(f"NO CHUNKS FOUND for: {question}")
        return (
            "I could not find relevant content in the books. \n\n"
            "Here is my answer based on general ML/DL knowledge:\n\n"
            + answer
        )

    sources = set([
        f"📖 {doc.metadata['source']} — Page {doc.metadata['page']}"
        for doc in source_docs
    ])

    source_text = "\n".join(sources)

    logging.info(f"SOURCES: {source_text}")

    return f"{answer}\n\n**Sources:**{source_text}"