import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable


def parse_and_clean_transcript(raw_text: str) -> Dict[str, str]:
    """Clean Thomson Reuters earning-call transcripts into prepared remarks and Q&A blocks."""
    pres_match = re.search(
        r"PRESENTATION SUMMARY(.*?)(?=QUESTIONS AND ANSWERS|$)",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    qa_match = re.search(
        r"QUESTIONS AND ANSWERS(.*)",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    def sanitize(text: str) -> str:
        if not text:
            return ""

        text = text.replace("\r", "\n")
        text = re.sub(r"<Sync\s+id=\"?L?\d+\"?\s*/>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\[[^\]]*\]", " ", text)
        text = re.sub(r"\(Operator Instructions\)|Operator\s*.*?(?=\n\n|\n[A-Z]|$)", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"^\s*(?:Operator|Conference|Analyst|Questioner|Moderator|Thank you|Hello|Good morning|Good afternoon|Good evening)\b.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r"^\s*[A-Z][A-Za-z0-9\s.,'\-]+\s+-\s+[A-Za-z\s]+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"(?im)^\s*(?:[IVX]+\.|[0-9]+\.)\s*", "", text)
        text = re.sub(r"(?im)^\s*(?:[IVX]+\.|[0-9]+\.)\s*(?:[A-Za-z0-9/&.,()\'-]+(?:\s+[A-Za-z0-9/&.,()\'-]+)*)?\s*$", "", text)
        text = re.sub(r"(?i)\b(?:Highlights|Results|Accomplishments|China|Summary|Overview|Financials|Business Review|iPhone|Mac|Apple Watch|Services|Q&A|Questions and Answers|PRESENTATION SUMMARY|QUESTIONS AND ANSWERS)\s*:\s*", "", text)
        text = re.sub(r"(?im)^\s*(?:Q|A)\s*:\s*", "", text)
        text = re.sub(r"(?im)^\s*(?:[A-Za-z][A-Za-z0-9\s/&.,()'-]*):\s*$", "", text)
        text = re.sub(r"^\s*[-*•=]+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"Can we have the next question.*?(?:\.|$)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    return {
        "prepared_remarks": sanitize(pres_match.group(1) if pres_match else ""),
        "qa_session": sanitize(qa_match.group(1) if qa_match else ""),
    }


def iter_transcript_files(root: Path) -> Iterable[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Transcript directory not found: {root}")
    yield from sorted(path for path in root.rglob("*.txt") if path.is_file())


def process_transcript_directory(input_dir: str, output_dir: str) -> list[Path]:
    source_dir = Path(input_dir)
    output_root = Path(output_dir)
    written_files: list[Path] = []

    for transcript_path in iter_transcript_files(source_dir):
        with transcript_path.open("r", encoding="utf-8", errors="ignore") as fh:
            cleaned = parse_and_clean_transcript(fh.read())

        relative_dir = transcript_path.relative_to(source_dir).parent
        target_dir = output_root / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / f"{transcript_path.stem}.clean.json"
        with target_path.open("w", encoding="utf-8") as out:
            json.dump(cleaned, out, ensure_ascii=False, indent=2)
        written_files.append(target_path)

    return written_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Thomson Reuters earnings transcript files for FinBERT sentiment modeling.")
    parser.add_argument("--input-dir", default="Raw data/Transcripts", help="Directory containing raw transcript text files.")
    parser.add_argument("--output-dir", default="Raw data/Transcripts/cleaned", help="Directory receiving cleaned JSON transcripts.")
    args = parser.parse_args()

    processed = process_transcript_directory(args.input_dir, args.output_dir)
    print(f"Cleaned {len(processed)} transcript files into {args.output_dir}")
