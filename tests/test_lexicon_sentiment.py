from pathlib import Path

from src.lexicon_sentiment import load_transcripts, score_loughran_mcdonald, score_vader


def test_load_transcripts_recurses_and_extracts_metadata(tmp_path: Path):
    ticker_dir = tmp_path / "AAPL"
    ticker_dir.mkdir()
    transcript_path = ticker_dir / "2016-Jan-26-AAPL.txt"
    transcript_path.write_text(
        """Prepared remarks:\n
We delivered strong revenue growth and excellent margins.
This quarter was very successful and we expect future profit.

Question and answer:
Analyst: Did margins fall?
Management: We continue to see strong demand.
""",
        encoding="utf-8",
    )

    df = load_transcripts(tmp_path)

    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"
    assert df.iloc[0]["filename"] == "2016-Jan-26-AAPL.txt"
    assert "prepared_remarks" in df.columns
    assert "strong" in df.iloc[0]["prepared_remarks"].lower()


def test_lexicon_scores_have_expected_direction():
    vader = score_vader("growth is strong and excellent")
    assert vader["compound"] > 0

    lm = score_loughran_mcdonald("liability, tax burden, loss and risk")
    assert lm["compound"] < 0
