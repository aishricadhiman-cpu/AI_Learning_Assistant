import re
from langchain_core.documents import Document
from typing import List
# cleaner.py

from spellchecker import SpellChecker

spell = SpellChecker()

def fix_spelling(text: str) -> str:
    """
    Fixes spelling mistakes in user query
    Preserves ML/DL technical terms that spellchecker doesn't know
    """
    # Technical ML/DL terms to never correct
    ML_TERMS = {
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
    }

    words = text.split()
    corrected = []

    for word in words:
        # Skip technical terms — never correct these
        if word.lower() in ML_TERMS:
            corrected.append(word)
            continue

        # Skip short words — spellchecker unreliable on these
        if len(word) <= 3:
            corrected.append(word)
            continue

        # Skip words with numbers like "layer1" or "conv2d"
        if any(char.isdigit() for char in word):
            corrected.append(word)
            continue

        # Fix spelling
        correction = spell.correction(word)

        # If no correction found keep original
        if correction is None:
            corrected.append(word)
        else:
            corrected.append(correction)

    return " ".join(corrected)

def remove_page_numbers(text: str) -> str:
    """
    removes standalone page numbers
    eg: 392,442
    """
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'—\s*\d+\s*—', '', text)
    text = re.sub(r'[Pp]age\s*\d+', '', text)
    text = re.sub(r'[Pp]g\.\s*\d+', '', text)
    return text


def clean_math(text):
    text = re.sub(r'\$.*?\$', '', text)          # inline math $x$
    text = re.sub(r'\$\$.*?\$\$', '', text)      # block math $$...$$
    text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', text) # latex commands
    text = re.sub(r'[=+\-*/^]{3,}', '', text)    # long symbol chains
    text = re.sub(r'\s+', ' ', text).strip()      # clean extra spaces
    return text

def remove_headers_footers(text: str) -> str:
    """
    Removes repetitive book headers and footers
    Examples removed:
    "Chapter 3 | Training Models"
    "Hands On Machine Learning"
    "Deep Learning with Python"
    """
    text = re.sub(r'Chapter\s*\d+\s*[|:]\s*.+', '', text)
    text = re.sub(r'Hands.?On Machine Learning.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Deep Learning with Python.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*CHAPTER\s*\d+.*$', '', text, flags=re.MULTILINE)
    return text


def remove_math(text: str) -> str:
    """
    Removes mathematical equations and symbols
    Keeps intuitive explanations, removes derivations
    Examples removed:
    "$x^2 + y^2$"  "$$\sigma(x) = 1/(1+e^{-x})$$"
    "∂L/∂w"  "Σ xi"
    """
    # LaTeX inline math
    text = re.sub(r'\$.*?\$', '', text)

    # LaTeX block math
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)

    # LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)

    # Greek letters and math symbols
    text = re.sub(r'[αβγδεζηθικλμνξπρστυφχψωΩΣΔ∂∇]', '', text)

    # Long symbol chains like === +++ ***
    text = re.sub(r'[=+\-*/^<>]{3,}', '', text)

    # Fractions and derivatives like ∂L/∂w
    text = re.sub(r'∂\w+/∂\w+', '', text)

    # Summation and product notation
    text = re.sub(r'[ΣΠ]\s*\w+', '', text)

    # Standalone numbers with operators like "= 0.5 +"
    text = re.sub(r'=\s*[\d.]+\s*[+\-*/]', '', text)

    return text


def remove_references(text: str) -> str:
    """
    Removes academic references and footnotes
    Examples removed:
    "[1]"  "(Hinton et al., 2012)"  "Figure 3-4"  "Table 2.1"
    """
    # Academic citations
    text = re.sub(r'\[[\d,\s]+\]', '', text)
    text = re.sub(r'\([A-Z][a-z]+\s+et\s+al\.,?\s*\d{4}\)', '', text)
    text = re.sub(r'\([A-Z][a-z]+,?\s*\d{4}\)', '', text)

    # Figure and table references
    text = re.sub(r'[Ff]igure\s*[\d-]+', '', text)
    text = re.sub(r'[Ff]ig\.\s*[\d-]+', '', text)
    text = re.sub(r'[Tt]able\s*[\d-]+', '', text)
    text = re.sub(r'[Ee]quation\s*[\d-]+', '', text)
    text = re.sub(r'[Ee]q\.\s*[\d-]+', '', text)

    # Footnote markers
    text = re.sub(r'\d+\s*↑', '', text)

    return text


def remove_noise(text: str) -> str:
    """
    Removes special characters, URLs, and other noise
    Examples removed:
    "http://..."  "www...."  "©"  "™"
    """
    # URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Copyright and trademark symbols
    text = re.sub(r'[©™®]', '', text)

    # Bullet point symbols
    text = re.sub(r'[•·▪▸►❯»]', '', text)

    # Repeated punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[-]{3,}', '', text)
    text = re.sub(r'[_]{3,}', '', text)

    # Non ASCII characters that slipped through
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    return text


def fix_formatting(text: str) -> str:
    """
    Fixes spacing, line breaks and general formatting
    """
    # Fix broken words across lines like "back-\npropagation"
    text = re.sub(r'-\n', '', text)

    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove excessive spaces
    text = re.sub(r' {2,}', ' ', text)

    # Fix spacing after punctuation
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

    # Strip each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Final strip
    text = text.strip()

    return text

def clean_text(text: str) -> str:
    """
    Runs all cleaning steps in correct order
    """
    text = remove_page_numbers(text)
    text = remove_headers_footers(text)
    text = remove_math(text)
    text = remove_references(text)
    text = remove_noise(text)
    text = fix_formatting(text)
    return text


# ── Apply to All Documents ─────────────────────────────────────
def clean_documents(docs: List[Document]) -> List[Document]:
    """
    Applies clean_text to every loaded document
    Filters out empty documents after cleaning
    """
    cleaned = []
    for doc in docs:
        cleaned_text = clean_text(doc.page_content)

        # Skip if page becomes empty after cleaning
        if len(cleaned_text.strip()) < 50:
            continue

        doc.page_content = cleaned_text
        cleaned.append(doc)

    print(f"Cleaned {len(cleaned)} pages successfully.")
    return cleaned

# ---Update Preprocess Query---------------------------------------

def preprocess_query(query: str) -> str:
    """
    Full query preprocessing pipeline:
    1. Strip whitespace
    2. Fix extra spaces
    3. Lowercase
    4. Fix spelling
    """
    query = query.strip()
    query = " ".join(query.split())   # remove extra spaces
    query = query.lower()             # normalize case
    query = fix_spelling(query)       # fix spelling mistakes
    return query


