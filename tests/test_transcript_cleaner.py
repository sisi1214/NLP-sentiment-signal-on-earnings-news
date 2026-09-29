import pathlib

from transcript_cleaner import parse_and_clean_transcript


SAMPLE = """
Thomson Reuters StreetEvents Event Brief
Q1 2016 Apple Inc Earnings Call

Corporate Participants
 * Luca Maestri
   Apple Inc. - CFO
 * Tim Cook
   Apple Inc. - CEO

PRESENTATION SUMMARY

We delivered a record quarter with revenue of $75.9 billion. Gross margin was 40.1%.

QUESTIONS AND ANSWERS

Operator Instructions

Tim Cook, Apple Inc. - CEO [3]

Q: We saw strong iPhone demand in China.
A: Demand was driven by our ecosystem and product quality.
Can we have the next question please.
"""


def test_parse_and_clean_transcript_extracts_sections_and_removes_noise():
    parsed = parse_and_clean_transcript(SAMPLE)

    prepared = parsed["prepared_remarks"].lower()
    qa = parsed["qa_session"].lower()

    assert "record quarter" in prepared
    assert "revenue of $75.9 billion" in prepared
    assert "gross margin" in prepared

    assert "strong iphone demand in china" in qa
    assert "demand was driven by our ecosystem" in qa
    assert "operator" not in qa
    assert "can we have the next question" not in qa
    assert "tim cook" not in qa
