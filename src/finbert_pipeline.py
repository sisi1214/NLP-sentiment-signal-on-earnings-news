from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.lexicon_sentiment import load_transcripts


device = "cpu"
MODEL_NAME = "ProsusAI/finbert"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
MODEL.eval()
LABELS = MODEL.config.id2label


def split_text_by_qna(text: str) -> tuple[str, str]:
    """Return (prepared_remarks, qna) blocks if Q&A markers are found."""
    if not text:
        return "", ""

    match = re.search(r"question and answer|questions and answers|q&a|q & a", text, flags=re.IGNORECASE)
    if not match:
        return text, ""

    prepared = text[: match.start()].strip()
    qna = text[match.start() :].strip()
    return prepared, qna


# Debugging helpers: limit noisy prints
DEBUG_QNA_PRINT_LIMIT = 10
_debug_qna_prints = 0


def chunk_text(text: str, max_tokens: int = 510, stride: int = 50) -> list[str]:
    """Split a long transcript into chunks that fit within BERT's token limit without triggering tokenizer warnings."""
    if not text or not text.strip():
        return [""]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        return [text]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        token_count = len(TOKENIZER.encode(candidate, add_special_tokens=False, truncation=True, max_length=max_tokens))

        if token_count <= max_tokens:
            current = candidate
            continue

        if current:
            chunks.append(current)
        sentence_token_count = len(TOKENIZER.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens))
        if sentence_token_count <= max_tokens:
            current = sentence
        else:
            words = sentence.split()
            current = ""
            temp = ""
            for word in words:
                next_text = word if not temp else f"{temp} {word}"
                next_count = len(TOKENIZER.encode(next_text, add_special_tokens=False, truncation=True, max_length=max_tokens))
                if next_count <= max_tokens:
                    temp = next_text
                else:
                    if temp:
                        chunks.append(temp)
                    temp = word
            if temp:
                current = temp

    if current:
        chunks.append(current)

    if not chunks:
        return [text]

    return chunks


@torch.no_grad()
def score_batch(texts: list[str], batch_size: int = 16) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = TOKENIZER(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        logits = MODEL(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        for p in probs:
            result = {LABELS[j]: float(p[j].item()) for j in range(len(LABELS))}
            results.append(result)
    return results


def score_long_document(text: str) -> dict[str, float]:
    """Average FinBERT probabilities across all chunks of a document."""
    chunks = chunk_text(text)
    chunk_scores = score_batch(chunks, batch_size=8)
    if not chunk_scores:
        return {label: 0.0 for label in LABELS.values()}

    averaged = {}
    for label in LABELS.values():
        averaged[label] = float(sum(score[label] for score in chunk_scores) / len(chunk_scores))
    averaged["compound"] = float(averaged.get("positive", 0.0) - averaged.get("negative", 0.0))
    return averaged


def score_finbert_document_with_qna(text: str, prepared_weight: float = 0.45, qna_weight: float = 0.55) -> dict[str, float]:
    prepared_text, qna_text = split_text_by_qna(text)
    global _debug_qna_prints
    if _debug_qna_prints < DEBUG_QNA_PRINT_LIMIT:
        try:
            print(f"DEBUG_QNA_LEN prepared={len(prepared_text)} qna={len(qna_text)}")
        except Exception:
            print("DEBUG_QNA_LEN prepared=<err> qna=<err>")
        _debug_qna_prints += 1
    prepared_scores = score_long_document(prepared_text) if prepared_text else {label: 0.0 for label in LABELS.values()}
    qna_scores = score_long_document(qna_text) if qna_text else {label: 0.0 for label in LABELS.values()}

    labels = sorted(set(prepared_scores) | set(qna_scores))
    combined: dict[str, float] = {}
    for label in labels:
        prepared_value = prepared_scores.get(label, 0.0)
        qna_value = qna_scores.get(label, 0.0)
        combined[label] = float((prepared_value * prepared_weight + qna_value * qna_weight) / (prepared_weight + qna_weight))
    combined["compound"] = float(combined.get("positive", 0.0) - combined.get("negative", 0.0))
    return combined


def parse_earnings_date(filename: str) -> pd.Timestamp | pd.NaT:
    filename = Path(filename).name
    for pattern in [
        r"(\d{4})[-_ ]([A-Za-z]{3})[-_ ](\d{1,2})[-_ ]([A-Z0-9]+)",
        r"(\d{4})(\d{2})(\d{2})",
    ]:
        match = re.search(pattern, filename, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            if len(match.groups()) == 4:
                year, month_name, day, _ = match.groups()
                month = pd.to_datetime(month_name, format="%b").month
                return pd.Timestamp(year=int(year), month=int(month), day=int(day))
            if len(match.groups()) == 3:
                year, month, day = match.groups()
                return pd.Timestamp(year=int(year), month=int(month), day=int(day))
        except (TypeError, ValueError):
            continue
    return pd.NaT


def get_forward_returns_for_ticker(ticker: str, earnings_date: pd.Timestamp, lookbacks: tuple[int, ...] = (1, 7, 28)) -> dict[str, float]:
    """Return forward returns for each lookback using the most recent available trading day before the earnings date as the anchor."""
    if pd.isna(earnings_date):
        return {f"ret_{offset}d": np.nan for offset in lookbacks}

    # sanitize ticker (strip leading $ and whitespace)
    ticker_str = str(ticker).strip()
    if ticker_str.startswith("$"):
        ticker_str = ticker_str.lstrip("$")

    # Try local CSV fallback from `Return classifcation/` to avoid repeated yfinance downloads
    local_dir = Path("Return classifcation")
    hist = None
    if local_dir.exists():
        # try common filename patterns like TICKER_2016_2020.csv
        candidates = list(local_dir.glob(f"{ticker_str}*.csv"))
        if not candidates:
            # try uppercase/lowercase variations
            candidates = list(local_dir.glob(f"{ticker_str.upper()}*.csv")) + list(local_dir.glob(f"{ticker_str.lower()}*.csv"))
        if candidates:
            try:
                hist = pd.read_csv(candidates[0], parse_dates=True, index_col=0)
                if "Close" not in hist.columns and "Adj Close" in hist.columns:
                    hist = hist.rename(columns={"Adj Close": "Close"})
                print(f"DEBUG_RET: loaded local returns for {ticker_str} from {candidates[0]}")
            except Exception as e:
                print(f"DEBUG_RET: failed to read local CSV {candidates[0]}: {e}")

    # If no local data, fall back to yfinance
    if hist is None or hist.empty:
        start = (earnings_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        end = (earnings_date + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        try:
            hist = yf.download(ticker_str, start=start, end=end, progress=False, auto_adjust=False)
        except Exception as e:
            print(f"DEBUG_RET: yf.download failed for {ticker_str} ({start} -> {end}): {e}")
            return {f"ret_{offset}d": np.nan for offset in lookbacks}
        if hist.empty:
            print(f"DEBUG_RET: no price data for {ticker_str} ({start} -> {end})")
            return {f"ret_{offset}d": np.nan for offset in lookbacks}

    # Normalize to have a Close column
    if "Close" not in hist.columns and "Adj Close" in hist.columns:
        hist = hist.rename(columns={"Adj Close": "Close"})
    hist = hist[["Close"]].dropna().copy()
    if hist.empty:
        return {f"ret_{offset}d": np.nan for offset in lookbacks}

    hist.index = pd.to_datetime(hist.index)
    anchor_date = hist.index[hist.index <= pd.Timestamp(earnings_date)]
    if len(anchor_date) == 0:
        baseline_idx = hist.index[0]
    else:
        baseline_idx = anchor_date[-1]

    baseline_close = float(hist.loc[baseline_idx, "Close"])
    if not np.isfinite(baseline_close) or baseline_close == 0:
        return {f"ret_{offset}d": np.nan for offset in lookbacks}

    trading_dates = hist.index.sort_values()
    future_dates = [d for d in trading_dates if d > baseline_idx]
    result: dict[str, float] = {}
    for offset in lookbacks:
        if len(future_dates) <= offset - 1:
            result[f"ret_{offset}d"] = np.nan
            continue
        target_date = future_dates[offset - 1]
        target_close = float(hist.loc[target_date, "Close"])
        result[f"ret_{offset}d"] = float(target_close / baseline_close - 1.0)
    return result


def build_finbert_sentiment_table(transcripts_df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    df = transcripts_df.copy()
    if limit is not None:
        df = df.head(limit).copy()

    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        # Use the full raw transcript so the Q&A splitter can detect Q&A sections.
        # `prepared_remarks` has already been extracted upstream and typically
        # excludes the Q&A, which prevents `score_finbert_document_with_qna`
        # from ever finding a Q&A block. Prefer `raw_text` first.
        text = row.get("raw_text") or row.get("prepared_remarks") or ""
        sentiment = score_finbert_document_with_qna(str(text))
        earnings_date = parse_earnings_date(str(row.get("filename", "")))
        rows.append(
            {
                "ticker": row.get("ticker"),
                "filename": row.get("filename"),
                "earnings_date": earnings_date,
                "prepared_remarks": row.get("prepared_remarks"),
                "sentiment_scores": sentiment,
                "finbert_positive": sentiment.get("positive", 0.0),
                "finbert_negative": sentiment.get("negative", 0.0),
                "finbert_neutral": sentiment.get("neutral", 0.0),
                "finbert_compound": sentiment.get("compound", 0.0),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["earnings_date"] = pd.to_datetime(out["earnings_date"])
    return out


def attach_forward_returns(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    if sentiment_df.empty:
        return sentiment_df

    merged_rows: list[dict[str, object]] = []
    for _, row in sentiment_df[["ticker", "earnings_date"]].drop_duplicates().iterrows():
        ticker = row["ticker"]
        earnings_date = row["earnings_date"]
        returns = get_forward_returns_for_ticker(ticker, earnings_date)
        merged_rows.append({"ticker": ticker, "earnings_date": earnings_date, **returns})

    returns_df = pd.DataFrame(merged_rows)
    returns_df["earnings_date"] = pd.to_datetime(returns_df["earnings_date"])
    return sentiment_df.merge(returns_df, on=["ticker", "earnings_date"], how="left")


def add_classification_labels(df: pd.DataFrame, return_col: str = "ret_1d") -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    out["finbert_signal"] = out["finbert_positive"] - out["finbert_negative"]
    out["ret_1d_label"] = np.where(out[return_col] > 0.0, 1, 0)
    out["buy_hold_sell"] = np.select(
        [out[return_col] < -0.02, out[return_col] > 0.02],
        ["sell", "buy"],
        default="hold",
    )
    return out


def quick_correlation_check(df: pd.DataFrame, x_col: str = "finbert_signal", y_col: str = "ret_1d") -> float:
    subset = df[[x_col, y_col]].dropna()
    if subset.empty:
        return np.nan
    return float(subset[x_col].corr(subset[y_col]))


def run_classifier(df: pd.DataFrame, target_col: str = "ret_1d_label") -> tuple[dict, dict]:
    feature_cols = ["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_compound"]
    X = df[feature_cols].fillna(0.0)
    y = df[target_col].fillna(0).astype(int)

    if y.nunique() < 2 or len(df) < 10:
        return {
            "status": "not_enough_data",
            "n_rows": int(len(df)),
            "n_unique_labels": int(y.nunique()),
        }, {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "classification_report": classification_report(y_test, preds, digits=4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return {"status": "ok", "n_rows": int(len(df))}, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinBERT sentiment scoring on transcript files and join forward returns.")
    parser.add_argument("--transcripts", type=str, default="Transcripts", help="Folder containing transcript text files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick testing on a few transcripts.")
    parser.add_argument("--output", type=str, default="finbert_transcript_scores.csv", help="CSV output path for the scored results.")
    args = parser.parse_args()

    transcripts_df = load_transcripts(args.transcripts)
    if transcripts_df.empty:
        raise ValueError(f"No transcript files found under: {args.transcripts}")

    sentiment_df = build_finbert_sentiment_table(transcripts_df, limit=args.limit)
    sentiment_df = attach_forward_returns(sentiment_df)
    sentiment_df = add_classification_labels(sentiment_df)

    # Print correlations for multiple horizons to check signal strength
    for col in ["ret_1d", "ret_7d", "ret_28d"]:
        corr_val = quick_correlation_check(sentiment_df, y_col=col)
        try:
            print(f"FinBERT signal vs {col} correlation: {corr_val:.4f}")
        except Exception:
            print(f"FinBERT signal vs {col} correlation: {corr_val}")

    status, metrics = run_classifier(sentiment_df)
    print(status)
    if metrics:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(metrics["classification_report"])

    output_path = Path(args.output)
    sentiment_df.to_csv(output_path, index=False)
    print(f"Saved {len(sentiment_df)} scored transcript rows to {output_path}")


if __name__ == "__main__":
    main()
