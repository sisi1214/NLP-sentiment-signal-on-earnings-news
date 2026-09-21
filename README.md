# NLP-sentiment-signal-on-earnings-news

Score sentiment from earnings call transcripts and test whether it predicts short-term equity returns.

This repository contains a compact pipeline to:

- extract prepared remarks and Q&A from earnings call transcripts
- score text with FinBERT and several lexicon-based methods
- attach forward returns (1d, 7d, 28d) from historical price data
- run quick classification and cross-validation experiments

Results: a straightforward FinBERT-only pipeline on prepared remarks (+ Q&A) did not produce a reliable predictive signal in this sample — correlations are near zero and a cross-validated classifier underperformed a majority-class baseline. See the **Results** section below.

## Table Of Contents

- Features
- Quick Start
- Usage
- Repository Layout
- Results (summary)
- Next steps
- License

## Features

- Transcript parsing (prepared remarks extraction, Q&A detection)
- FinBERT scoring (chunking, batch inference)
- Lexicon scoring (Loughran-McDonald, VADER)
- Forward-return attachment using local CSVs (fallback to yfinance)
- Classification and cross-validation utilities

## Quick Start

1. Create a Python environment and install dependencies (use your preferred tool):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Score transcripts and save results:

```bash
python src/finbert_pipeline.py --transcripts Transcripts --output finbert_transcript_scores.csv
```

3. Run 5-fold cross-validation on the scored features (`ret_7d_label`):

```bash
python src/finbert_pipeline.py --transcripts Transcripts --cv
```

## Usage

- `src/preprocess_transcripts.py` — extract and preprocess prepared remarks (saves vectorizer artifacts to `Transcripts/processed/`).
- `src/lexicon_sentiment.py` — load transcripts and compute lexicon scores.
- `src/finbert_pipeline.py` — run FinBERT scoring, attach returns, compute labels, run classifiers and CV. See flags: `--limit` for quick tests and `--cv` for cross-validation.

## Repository Layout

- `Transcripts/` — raw transcript files organized by ticker
- `Return classifcation/` — local CSVs of historical prices used as primary return source
- `src/` — pipeline source code (`finbert_pipeline.py`, `lexicon_sentiment.py`, `preprocess_transcripts.py`)
- `tests/` — unit tests (small)

## Results (short summary)

- Full dataset size: ~188 scored transcripts (example run saved to `finbert_transcript_scores.csv`).
- Correlations between FinBERT compound signal and forward returns were near zero (example full-run values: ret_1d ≈ -0.046, ret_7d ≈ -0.027, ret_28d ≈ -0.053).
- 5-fold stratified CV with a balanced Logistic Regression on `ret_7d_label`: mean accuracy ≈ 0.489 (std ≈ 0.052). Majority-class baseline ≈ 0.561 — the classifier did not beat the baseline in this setup.

## Next steps (ideas)

- Add more features: lexicon scores, trading volume/volatility, simple technicals, or metadata (sector, market cap).
- Try regression on `ret_7d` (Ridge) and bucket predictions rather than directly classifying binary labels.
- Improve label construction (different thresholds, multi-class buy/hold/sell), or tune class balancing (upsampling/SMOTE).
- Explore time-aware validation (rolling-window CV) and transaction-cost-aware backtesting if you move to a signal-based strategy.

