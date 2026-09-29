import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from transcript_cleaner import parse_and_clean_transcript


LM_DICTIONARIES = {
    "Negative": {
        "loss", "decline", "declined", "declining", "negative", "weak", "weakness", "risk", "risks",
        "problem", "problems", "challenge", "challenges", "difficult", "difficulty", "slowing",
        "slower", "impairment", "uncertain", "uncertainty", "downturn", "recession", "volatility", "pressure"
    },
    "Positive": {
        "growth", "strong", "strength", "increase", "increased", "improve", "improved", "improvement",
        "benefit", "benefits", "good", "excellent", "record", "confidence", "favorable", "opportunity",
        "positive", "outperform", "outperformed", "profit", "profits", "revenue", "stable", "growth", "expanded"
    },
    "Uncertainty": {
        "uncertain", "uncertainty", "possible", "perhaps", "may", "might", "could", "would", "likely",
        "approximate", "approximately", "depends", "depend", "assume", "assumes", "appears", "seems",
        "expected", "estimate", "estimates", "indicate", "indicates", "potentially", "possibly"
    },
    "Litigious": {
        "lawsuit", "lawsuits", "litigation", "claim", "claims", "liability", "liabilities", "legal",
        "regulatory", "regulation", "compliance", "settlement", "court", "penalty", "penalties", "appeal"
    },
    "Constraining": {
        "must", "required", "require", "requires", "cannot", "can\'t", "limit", "limited", "restrict",
        "restriction", "constraints", "constrained", "depend", "depends", "conditional", "only"
    },
}


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def tokenize_words(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def count_syllables(word: str) -> int:
    if not word:
        return 0
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    vowels = "aeiouy"
    groups = re.findall(r"[aeiouy]+", word)
    count = sum(len(group) for group in groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def parse_transcript_metadata(path: Path) -> Dict[str, str]:
    path = Path(path)
    if path.suffix == ".json":
        stem = path.stem.replace(".clean", "")
    else:
        stem = path.stem

    if "-" not in stem:
        raise ValueError(f"Unexpected transcript naming convention: {path}")

    base_name = stem.rsplit("-", 1)
    if len(base_name) != 2:
        raise ValueError(f"Unable to parse ticker/date from transcript name: {path}")

    date_part, ticker = base_name
    date_value = datetime.strptime(date_part, "%Y-%b-%d").strftime("%Y-%m-%d")
    quarter = (datetime.strptime(date_value, "%Y-%m-%d").month - 1) // 3 + 1
    return {
        "ticker": ticker.upper(),
        "date": date_value,
        "quarter": f"{datetime.strptime(date_value, '%Y-%m-%d').year}-Q{quarter}",
    }


def compute_lm_dictionary_features(text: str) -> Dict[str, float]:
    tokens = tokenize_words(text)
    total_words = len(tokens)
    counts = {category: 0 for category in LM_DICTIONARIES}

    if total_words:
        token_counter = Counter(tokens)
        for category, words in LM_DICTIONARIES.items():
            counts[category] = sum(token_counter[word] for word in words if word in token_counter)

    return {
        f"LM_Freq_{category}": safe_divide(counts[category], total_words)
        for category in LM_DICTIONARIES
    }


def compute_text_complexity_metrics(text: str) -> Dict[str, float]:
    tokens = tokenize_words(text)
    word_count = len(tokens)
    if not text or word_count == 0:
        return {
            "Word_Count": 0,
            "Sentence_Count": 0,
            "TTR": 0.0,
            "Fog_Index": 0.0,
            "Complex_Word_Count": 0,
        }

    sentences = [segment.strip() for segment in re.split(r"[.!?]+", text) if segment.strip()]
    sentence_count = max(len(sentences), 1)
    unique_tokens = len(set(tokens))
    complex_word_count = sum(1 for token in tokens if count_syllables(token) >= 3)
    ttr = safe_divide(unique_tokens, word_count)
    fog_index = 0.4 * ((safe_divide(word_count, sentence_count)) + 100 * safe_divide(complex_word_count, word_count))

    return {
        "Word_Count": word_count,
        "Sentence_Count": sentence_count,
        "TTR": ttr,
        "Fog_Index": fog_index,
        "Complex_Word_Count": complex_word_count,
    }


def _finbert_heuristic_vector(text: str) -> List[float]:
    tokens = tokenize_words(text)
    token_counter = Counter(tokens)
    positive = sum(token_counter[word] for word in LM_DICTIONARIES["Positive"])
    negative = sum(token_counter[word] for word in LM_DICTIONARIES["Negative"])
    neutral = max(1, len(tokens) - positive - negative)
    vector = [float(positive), float(negative), float(neutral)]
    total = sum(vector)
    if total == 0:
        return [0.33, 0.33, 0.34]
    return [value / total for value in vector]


def compute_finbert_probabilities(text: str) -> List[float]:
    if not text or not text.strip():
        return [0.0, 0.0, 1.0]
    return _finbert_heuristic_vector(text)


def build_transcript_feature_row(cleaned_text: Dict[str, str]) -> Dict[str, float]:
    prepared = cleaned_text.get("prepared_remarks", "")
    qa = cleaned_text.get("qa_session", "")
    prepared_probs = compute_finbert_probabilities(prepared)
    qa_probs = compute_finbert_probabilities(qa)
    prepared_metrics = compute_text_complexity_metrics(prepared)
    qa_metrics = compute_text_complexity_metrics(qa)
    overall_metrics = compute_text_complexity_metrics(f"{prepared} {qa}")
    prepared_lm = compute_lm_dictionary_features(prepared)
    qa_lm = compute_lm_dictionary_features(qa)

    row = {
        "Prepared_Word_Count": prepared_metrics["Word_Count"],
        "QA_Word_Count": qa_metrics["Word_Count"],
        "Total_Word_Count": overall_metrics["Word_Count"],
        "Prepared_TTR": prepared_metrics["TTR"],
        "QA_TTR": qa_metrics["TTR"],
        "Overall_TTR": overall_metrics["TTR"],
        "QA_to_Prepared_Word_Ratio": safe_divide(qa_metrics["Word_Count"], prepared_metrics["Word_Count"] + 1e-5),
        "Prepared_Fog_Index": prepared_metrics["Fog_Index"],
        "QA_Fog_Index": qa_metrics["Fog_Index"],
        "Overall_Fog_Index": overall_metrics["Fog_Index"],
        "FinBERT_Positive_Prepared": prepared_probs[0],
        "FinBERT_Negative_Prepared": prepared_probs[1],
        "FinBERT_Neutral_Prepared": prepared_probs[2],
        "FinBERT_Positive_QA": qa_probs[0],
        "FinBERT_Negative_QA": qa_probs[1],
        "FinBERT_Neutral_QA": qa_probs[2],
        "Sentiment_Delta_Positive": qa_probs[0] - prepared_probs[0],
        "QA_Uncertainty_Ratio": qa_lm["LM_Freq_Uncertainty"],
    }
    row.update(prepared_lm)
    row.update({f"QA_{key}": value for key, value in qa_lm.items()})
    return row


def load_return_data(return_dir: Path) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(Path(return_dir).glob("*.csv")):
        df = pd.read_csv(csv_path)
        if "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        ticker = csv_path.stem.split("_")[0].upper()
        frames[ticker] = df.sort_values("Date").reset_index(drop=True)
    return frames


def align_stock_return(row: Dict[str, str], stock_df: pd.DataFrame) -> Dict[str, float]:
    event_date = pd.to_datetime(row["date"])
    if stock_df.empty:
        return {"return_1d": np.nan, "return_3d": np.nan, "return_classification": np.nan}

    event_mask = stock_df["Date"] == event_date
    if not event_mask.any():
        return {"return_1d": np.nan, "return_3d": np.nan, "return_classification": np.nan}

    event_idx = stock_df.index[event_mask][0]
    close_today = float(stock_df.loc[event_idx, "Close"])
    close_1d = float(stock_df.loc[event_idx + 1, "Close"]) if event_idx + 1 < len(stock_df) else np.nan
    close_3d = float(stock_df.loc[event_idx + 3, "Close"]) if event_idx + 3 < len(stock_df) else np.nan

    ret_1d = safe_divide(close_1d - close_today, close_today) if pd.notna(close_today) and pd.notna(close_1d) else np.nan
    ret_3d = safe_divide(close_3d - close_today, close_today) if pd.notna(close_today) and pd.notna(close_3d) else np.nan
    label = 1 if pd.notna(ret_3d) and ret_3d >= 0 else 0 if pd.notna(ret_3d) else np.nan
    return {"return_1d": ret_1d, "return_3d": ret_3d, "return_classification": label}


def iter_cleaned_transcripts(root: Path) -> Iterable[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Transcript directory not found: {root}")
    yield from sorted(path for path in root.rglob("*.clean.json") if path.is_file())


def engineer_feature_dataset(transcript_dir: str, return_dir: str, output_path: str) -> pd.DataFrame:
    transcript_root = Path(transcript_dir)
    return_root = Path(return_dir)
    stock_frames = load_return_data(return_root)
    rows: List[Dict[str, object]] = []

    for transcript_path in iter_cleaned_transcripts(transcript_root):
        metadata = parse_transcript_metadata(transcript_path)
        ticker = metadata["ticker"]
        if ticker not in stock_frames:
            continue

        with transcript_path.open("r", encoding="utf-8") as fh:
            cleaned = json.load(fh)
        feature_row = build_transcript_feature_row(cleaned)
        feature_row.update({
            "ticker": ticker,
            "date": metadata["date"],
            "quarter": metadata["quarter"],
        })
        feature_row.update(align_stock_return(metadata, stock_frames[ticker]))
        rows.append(feature_row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engineer sentiment and text metrics from cleaned earnings-call transcripts.")
    parser.add_argument("--transcript-dir", default="Raw data/Transcripts/cleaned", help="Directory with cleaned JSON transcript files.")
    parser.add_argument("--return-dir", default="Raw data/Return classifcation", help="Directory with per-ticker stock return CSV files.")
    parser.add_argument("--output", default="Raw data/Transcripts/processed/engineered_features.csv", help="Output CSV path for engineered features.")
    args = parser.parse_args()

    dataset = engineer_feature_dataset(args.transcript_dir, args.return_dir, args.output)
    print(f"Engineered {len(dataset)} transcript rows into {args.output}")
