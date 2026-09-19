import os
import re
import pickle
from pathlib import Path

import nltk
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import scipy.sparse as sp


DATA_DIR = Path("Transcripts")
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_prepared_remarks(text: str) -> str:
    # Find the prepared remarks section and exclude the Q&A
    start_pattern = re.compile(r"prepared remarks[:\s\-]*|presentation[:\s\-]*", flags=re.IGNORECASE)
    end_patterns = [
        r"question and answer",
        r"question-and-answer",
        r"questions and answers",
        r"q&a",
        r"q & a",
        r"operator",
        r"thank you and operator",
    ]

    m = start_pattern.search(text)
    if not m:
        # fallback: if no explicit section, try to take everything until Q&A markers
        start_idx = 0
    else:
        start_idx = m.end()

    end_idx = None
    for ep in end_patterns:
        rm = re.search(ep, text[start_idx:], flags=re.IGNORECASE)
        if rm:
            candidate = start_idx + rm.start()
            if end_idx is None or candidate < end_idx:
                end_idx = candidate

    if end_idx is None:
        extracted = text[start_idx:]
    else:
        extracted = text[start_idx:end_idx]

    # Remove speaker labels and timestamps like "Operator:" or "John Doe - CEO"
    extracted = re.sub(r"^\s*[A-Z][A-Za-z0-9\-\.,' ]{0,60}:", "", extracted, flags=re.M)
    # Remove parentheses contents (asides)
    extracted = re.sub(r"\(.*?\)", " ", extracted, flags=re.S)
    return extracted.strip()



def preprocess_text(text: str, stemmer, stop_words) -> str:
    text = text.lower()
    # remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)
    # simple whitespace tokenizer
    tokens = [t for t in text.split() if t.isalpha() and len(t) > 2]
    tokens = [t for t in tokens if t not in stop_words]
    stems = [stemmer.stem(w) for w in tokens]
    return " ".join(stems)


def main():
    # use sklearn's built-in stop words and NLTK Porter stemmer (no data downloads)
    stop_words = set(ENGLISH_STOP_WORDS)
    stemmer = PorterStemmer()

    docs = []
    original_texts = []
    file_names = []

    if not DATA_DIR.exists():
        print(f"Transcripts directory not found at {DATA_DIR}.")
        return

    for p in sorted(DATA_DIR.rglob("*.txt")):
        if not p.is_file() or p.name.startswith('.'):
            continue
        if 'processed' in p.parts:
            continue
        try:
            text = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        extracted = extract_prepared_remarks(text)
        if not extracted or len(extracted) < 20:
            # skip tiny extracts
            continue
        processed = preprocess_text(extracted, stemmer, stop_words)
        if not processed:
            continue
        docs.append(processed)
        original_texts.append(extracted)
        file_names.append(p.name)

    if not docs:
        print("No prepared remarks extracted from transcripts.")
        return

    # split (handle very small datasets)
    if len(docs) < 2:
        X_train = docs
        X_test = []
        fn_train = file_names
        fn_test = []
    else:
        X_train, X_test, y_train, y_test, fn_train, fn_test = train_test_split(
            docs, [None]*len(docs), file_names, test_size=0.2, random_state=42
        )

    # Vectorizers
    count_vect = CountVectorizer()
    tfidf_vect = TfidfVectorizer()

    X_train_count = count_vect.fit_transform(X_train)
    if X_test:
        X_test_count = count_vect.transform(X_test)
    else:
        X_test_count = sp.csr_matrix((0, X_train_count.shape[1]))

    X_train_tfidf = tfidf_vect.fit_transform(X_train)
    if X_test:
        X_test_tfidf = tfidf_vect.transform(X_test)
    else:
        X_test_tfidf = sp.csr_matrix((0, X_train_tfidf.shape[1]))

    # Save artifacts
    with open(OUT_DIR / 'count_vectorizer.pkl', 'wb') as f:
        pickle.dump(count_vect, f)
    with open(OUT_DIR / 'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vect, f)

    sp.save_npz(OUT_DIR / 'X_train_count.npz', X_train_count)
    sp.save_npz(OUT_DIR / 'X_test_count.npz', X_test_count)
    sp.save_npz(OUT_DIR / 'X_train_tfidf.npz', X_train_tfidf)
    sp.save_npz(OUT_DIR / 'X_test_tfidf.npz', X_test_tfidf)

    # Save metadata
    with open(OUT_DIR / 'filenames_train.pkl', 'wb') as f:
        pickle.dump(fn_train, f)
    with open(OUT_DIR / 'filenames_test.pkl', 'wb') as f:
        pickle.dump(fn_test, f)

    print(f"Processed {len(docs)} documents.")
    print(f"Train/test sizes: {len(X_train)} / {len(X_test)}")
    print(f"Saved artifacts to {OUT_DIR}")


if __name__ == '__main__':
    main()
