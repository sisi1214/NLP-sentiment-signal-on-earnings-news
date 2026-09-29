from pathlib import Path

from feature_engineering import (
    compute_lm_dictionary_features,
    compute_text_complexity_metrics,
    parse_transcript_metadata,
)


def test_parse_transcript_metadata_extracts_ticker_and_quarter():
    path = Path("Raw data/Transcripts/AAPL/2016-Jan-26-AAPL.txt")
    parsed = parse_transcript_metadata(path)

    assert parsed["ticker"] == "AAPL"
    assert parsed["date"] == "2016-01-26"
    assert parsed["quarter"] == "2016-Q1"


def test_lm_dictionary_and_complexity_metrics_are_numeric():
    text = "Risk and uncertainty may limit growth, but positive results improve confidence."

    lm = compute_lm_dictionary_features(text)
    complexity = compute_text_complexity_metrics(text)

    assert lm["LM_Freq_Negative"] >= 0
    assert lm["LM_Freq_Positive"] >= 0
    assert lm["LM_Freq_Uncertainty"] >= 0
    assert complexity["TTR"] >= 0
    assert complexity["Fog_Index"] >= 0
