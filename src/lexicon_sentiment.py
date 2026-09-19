from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_NEGATIVE_WORDS = {
    "adverse",
    "burden",
    "challenge",
    "challenges",
    "concern",
    "concerns",
    "cost",
    "costs",
    "decline",
    "declined",
    "decrease",
    "difficult",
    "difficulties",
    "disappoint",
    "disappointing",
    "exposure",
    "fraud",
    "impairment",
    "issue",
    "issues",
    "lawsuit",
    "liability",
    "liabilities",
    "litigation",
    "loss",
    "losses",
    "penalty",
    "risk",
    "risks",
    "tax",
    "uncertainty",
    "weak",
    "weakness",
    "writeoff",
    "write-off",
}

_POSITIVE_WORDS = {
    "benefit",
    "benefits",
    "efficiency",
    "excellent",
    "gain",
    "gains",
    "growth",
    "improve",
    "improved",
    "increase",
    "increased",
    "opportunity",
    "opportunities",
    "profit",
    "profits",
    "revenue",
    "strong",
    "success",
    "successful",
    "favorable",
    "higher",
}


def extract_prepared_remarks(text: str) -> str:
    """Keep the prepared remarks section and drop Q&A and speaker annotations."""
    start_pattern = re.compile(r"prepared remarks[:\s\-]*|presentation[:\s\-]*|overview[:\s\-]*", flags=re.IGNORECASE)
    match = start_pattern.search(text)
    start_idx = match.end() if match else 0

    end_patterns = [
        r"question and answer",
        r"question-and-answer",
        r"questions and answers",
        r"q&a",
        r"q & a",
        r"operator",
        r"thank you and operator",
    ]

    end_idx = None
    for pattern in end_patterns:
        found = re.search(pattern, text[start_idx:], flags=re.IGNORECASE)
        if found:
            candidate = start_idx + found.start()
            if end_idx is None or candidate < end_idx:
                end_idx = candidate

    extracted = text[start_idx:] if end_idx is None else text[start_idx:end_idx]

    extracted = re.sub(r"^\s*[A-Z][A-Za-z0-9\-\.,' ]{0,60}:", "", extracted, flags=re.MULTILINE)
    extracted = re.sub(r"\(.*?\)", " ", extracted, flags=re.DOTALL)
    extracted = re.sub(r"\s+", " ", extracted).strip()
    return extracted


def _iter_transcript_files(root: str | Path) -> Iterable[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Transcript directory not found: {root_path}")

    for path in sorted(root_path.rglob("*.txt")):
        if path.is_file() and not path.name.startswith("."):
            yield path


def load_transcripts(transcripts_dir: str | Path = "Transcripts") -> pd.DataFrame:
    """Load all transcript txt files under the Transcript directory tree into a DataFrame."""
    rows: list[dict[str, str]] = []

    for path in _iter_transcript_files(transcripts_dir):
        if path.parent.name == "processed":
            continue

        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        prepared = extract_prepared_remarks(raw_text)
        if not prepared:
            continue

        ticker = path.parent.name
        if not ticker or ticker == "Transcripts":
            ticker_match = re.search(r"([A-Z]{2,10})\.[Tt][Xx][Tt]$|([A-Z]{2,10})\.txt$", path.name)
            ticker = ticker_match.group(1) or ticker_match.group(2) if ticker_match else "UNKNOWN"

        rows.append(
            {
                "ticker": ticker,
                "filename": path.name,
                "transcript_path": str(path),
                "prepared_remarks": prepared,
                "raw_text": raw_text,
            }
        )

    return pd.DataFrame(rows, columns=["ticker", "filename", "transcript_path", "prepared_remarks", "raw_text"])


def score_vader(text: str) -> dict[str, float]:
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def score_loughran_mcdonald(text: str) -> dict[str, float]:
    tokens = _tokenize(text)
    if not tokens:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "compound": 0.0}

    positive_hits = sum(1 for token in tokens if token in _POSITIVE_WORDS)
    negative_hits = sum(1 for token in tokens if token in _NEGATIVE_WORDS)
    total_hits = positive_hits + negative_hits

    if total_hits == 0:
        compound = 0.0
    else:
        compound = (positive_hits - negative_hits) / total_hits

    return {
        "positive": float(positive_hits),
        "negative": float(negative_hits),
        "neutral": float(max(0.0, len(tokens) - total_hits)),
        "compound": float(compound),
    }


def main() -> None:
    df = load_transcripts("Transcripts")
    if df.empty:
        print("No transcript files were loaded from Transcripts.")
        return

    df["vader_score"] = df["prepared_remarks"].apply(score_vader)
    df["lm_score"] = df["prepared_remarks"].apply(score_loughran_mcdonald)

    print(f"Loaded {len(df)} transcripts.")
    print(df[["ticker", "filename", "vader_score", "lm_score"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
