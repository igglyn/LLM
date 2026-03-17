"""
Corpus management — fetch, cache, and export training data.

Sources:
    - Project Gutenberg (direct fetch)
    - HuggingFace datasets (swappable)

Storage:
    - SQLite metadata + document index
    - Raw .txt files on disk
    - JSONL export for structured consumption
"""

import os
import re
import json
import time
import hashlib
import sqlite3
import logging
import urllib.request
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass, asdict

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GUTENBERG_MIRROR = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
GUTENBERG_CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"

DEFAULT_DATA_DIR = Path("data/corpus")
DEFAULT_DB_PATH  = DEFAULT_DATA_DIR / "corpus.db"
DEFAULT_RAW_DIR  = DEFAULT_DATA_DIR / "raw"
DEFAULT_JSONL_DIR = DEFAULT_DATA_DIR / "jsonl"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Document:
    doc_id:    str          # sha256 of content
    source:    str          # "gutenberg", "hf", etc.
    title:     str
    author:    str
    language:  str
    raw_path:  str          # path to raw .txt
    byte_len:  int
    fetched_at: int         # unix timestamp


# ---------------------------------------------------------------------------
# SQLite layer
# ---------------------------------------------------------------------------

class CorpusDB:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id      TEXT PRIMARY KEY,
                source      TEXT NOT NULL,
                title       TEXT NOT NULL,
                author      TEXT NOT NULL,
                language    TEXT NOT NULL,
                raw_path    TEXT NOT NULL,
                byte_len    INTEGER NOT NULL,
                fetched_at  INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fetch_log (
                url         TEXT PRIMARY KEY,
                status      TEXT NOT NULL,
                fetched_at  INTEGER NOT NULL,
                error       TEXT
            );
        """)
        self.conn.commit()

    def insert_document(self, doc: Document):
        self.conn.execute("""
            INSERT OR IGNORE INTO documents
            VALUES (:doc_id, :source, :title, :author, :language,
                    :raw_path, :byte_len, :fetched_at)
        """, asdict(doc))
        self.conn.commit()

    def document_exists(self, doc_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row is not None

    def log_fetch(self, url: str, status: str, error: Optional[str] = None):
        self.conn.execute("""
            INSERT OR REPLACE INTO fetch_log VALUES (?, ?, ?, ?)
        """, (url, status, int(time.time()), error))
        self.conn.commit()

    def all_documents(self) -> list[Document]:
        rows = self.conn.execute("SELECT * FROM documents").fetchall()
        cols = [c[1] for c in self.conn.execute("PRAGMA table_info(documents)").fetchall()]
        return [Document(**dict(zip(cols, row))) for row in rows]

    def total_bytes(self) -> int:
        row = self.conn.execute("SELECT SUM(byte_len) FROM documents").fetchone()
        return row[0] or 0

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def _fetch_url(url: str, retries: int = 3, delay: float = 2.0) -> Optional[bytes]:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "corpus-fetcher/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            log.warning(f"Fetch attempt {attempt+1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    return None


def _clean_gutenberg(raw: bytes) -> str:
    """Strip Gutenberg header/footer boilerplate."""
    text = raw.decode("utf-8", errors="replace")

    # Find content start
    start_markers = [
        r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG",
        r"\*\*\*START OF (THE|THIS) PROJECT GUTENBERG",
    ]
    for marker in start_markers:
        m = re.search(marker, text, re.IGNORECASE)
        if m:
            text = text[m.end():]
            # Skip the rest of the start line
            text = text[text.find("\n")+1:]
            break

    # Find content end
    end_markers = [
        r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG",
        r"\*\*\*END OF (THE|THIS) PROJECT GUTENBERG",
    ]
    for marker in end_markers:
        m = re.search(marker, text, re.IGNORECASE)
        if m:
            text = text[:m.start()]
            break

    return text.strip()


def fetch_gutenberg(
    book_ids: list[int],
    db: CorpusDB,
    raw_dir: Path = DEFAULT_RAW_DIR,
    language_filter: str = "en",
) -> int:
    """
    Fetch books from Project Gutenberg by ID.
    Returns number of new documents added.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    added = 0

    for book_id in book_ids:
        url = GUTENBERG_MIRROR.format(id=book_id)
        raw = _fetch_url(url)

        if raw is None:
            db.log_fetch(url, "failed")
            log.warning(f"Could not fetch book {book_id}")
            continue

        try:
            text = _clean_gutenberg(raw)
        except Exception as e:
            db.log_fetch(url, "parse_error", str(e))
            continue

        if not text or len(text) < 1000:
            db.log_fetch(url, "too_short")
            continue

        content_bytes = text.encode("utf-8")
        doc_id = hashlib.sha256(content_bytes).hexdigest()

        if db.document_exists(doc_id):
            log.info(f"Book {book_id} already in corpus, skipping")
            db.log_fetch(url, "duplicate")
            continue

        raw_path = raw_dir / f"{doc_id[:16]}.txt"
        raw_path.write_bytes(content_bytes)

        # Best-effort metadata extraction
        title  = _extract_gutenberg_meta(text, "Title") or f"gutenberg_{book_id}"
        author = _extract_gutenberg_meta(text, "Author") or "Unknown"

        doc = Document(
            doc_id=doc_id,
            source="gutenberg",
            title=title,
            author=author,
            language=language_filter,
            raw_path=str(raw_path),
            byte_len=len(content_bytes),
            fetched_at=int(time.time()),
        )
        db.insert_document(doc)
        db.log_fetch(url, "ok")
        log.info(f"Added book {book_id}: {title} ({len(content_bytes)//1024}KB)")
        added += 1

    return added


def _extract_gutenberg_meta(text: str, field: str) -> Optional[str]:
    m = re.search(rf"^{field}:\s*(.+)$", text[:2000], re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip() if m else None


def fetch_hf_dataset(
    dataset_name: str,
    text_column: str,
    db: CorpusDB,
    raw_dir: Path = DEFAULT_RAW_DIR,
    split: str = "train",
    max_docs: Optional[int] = None,
) -> int:
    """
    Load documents from a HuggingFace dataset.
    Swappable alternative to Gutenberg fetcher.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    added = 0

    for i, record in enumerate(dataset):
        if max_docs and i >= max_docs:
            break

        text = record.get(text_column, "")
        if not text or len(text) < 500:
            continue

        content_bytes = text.encode("utf-8")
        doc_id = hashlib.sha256(content_bytes).hexdigest()

        if db.document_exists(doc_id):
            continue

        raw_path = raw_dir / f"{doc_id[:16]}.txt"
        raw_path.write_bytes(content_bytes)

        doc = Document(
            doc_id=doc_id,
            source=dataset_name,
            title=record.get("title", f"doc_{i}"),
            author=record.get("author", "Unknown"),
            language=record.get("language", "en"),
            raw_path=str(raw_path),
            byte_len=len(content_bytes),
            fetched_at=int(time.time()),
        )
        db.insert_document(doc)
        added += 1

    log.info(f"Added {added} documents from {dataset_name}")
    return added


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

def export_raw_text(
    db: CorpusDB,
    out_path: Path,
    separator: str = "\n\n---\n\n",
):
    """
    Concatenate all corpus documents into a single raw .txt file.
    Used as input to byte-level patchers.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    docs = db.all_documents()

    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            try:
                text = Path(doc.raw_path).read_text(encoding="utf-8", errors="replace")
                f.write(text)
                f.write(separator)
            except Exception as e:
                log.warning(f"Could not read {doc.raw_path}: {e}")

    total_mb = out_path.stat().st_size / 1e6
    log.info(f"Exported raw text: {out_path} ({total_mb:.1f} MB)")


def export_jsonl(
    db: CorpusDB,
    out_path: Path,
    chunk_size: int = 4096,
):
    """
    Export corpus as JSONL with byte chunks.
    Each record: {doc_id, title, author, source, chunk_idx, text}
    Used for structured training consumption.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    docs = db.all_documents()
    total_records = 0

    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            try:
                text = Path(doc.raw_path).read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                log.warning(f"Could not read {doc.raw_path}: {e}")
                continue

            # Chunk by character count (approximate byte boundary)
            for chunk_idx, start in enumerate(range(0, len(text), chunk_size)):
                chunk = text[start:start + chunk_size]
                if not chunk.strip():
                    continue
                record = {
                    "doc_id":    doc.doc_id,
                    "title":     doc.title,
                    "author":    doc.author,
                    "source":    doc.source,
                    "chunk_idx": chunk_idx,
                    "text":      chunk,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_records += 1

    log.info(f"Exported JSONL: {out_path} ({total_records} records)")


# ---------------------------------------------------------------------------
# Corpus iterator (for training)
# ---------------------------------------------------------------------------

def iter_raw_bytes(
    db: CorpusDB,
    sequence_len: int = 512,
    shuffle: bool = True,
) -> Iterator[bytes]:
    """
    Yield fixed-length byte sequences from the corpus.
    Used directly by the dataloader.
    """
    import random
    docs = db.all_documents()
    if shuffle:
        random.shuffle(docs)

    buffer = b""
    for doc in docs:
        try:
            content = Path(doc.raw_path).read_bytes()
        except Exception:
            continue

        buffer += content

        while len(buffer) >= sequence_len:
            yield buffer[:sequence_len]
            buffer = buffer[sequence_len:]

    # Yield remaining with padding
    if buffer:
        padded = buffer + b"\x00" * (sequence_len - len(buffer))
        yield padded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Corpus management")
    sub = parser.add_subparsers(dest="cmd")

    # fetch gutenberg
    p_fetch = sub.add_parser("fetch", help="Fetch Gutenberg books")
    p_fetch.add_argument("ids", nargs="+", type=int, help="Book IDs")

    # export
    p_export = sub.add_parser("export", help="Export corpus")
    p_export.add_argument("--raw", type=Path, default=DEFAULT_JSONL_DIR / "corpus.txt")
    p_export.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL_DIR / "corpus.jsonl")

    # stats
    sub.add_parser("stats", help="Show corpus stats")

    args = parser.parse_args()
    db = CorpusDB()

    if args.cmd == "fetch":
        n = fetch_gutenberg(args.ids, db)
        print(f"Added {n} documents. Total corpus: {db.total_bytes()/1e6:.1f} MB")

    elif args.cmd == "export":
        export_raw_text(db, args.raw)
        export_jsonl(db, args.jsonl)

    elif args.cmd == "stats":
        docs = db.all_documents()
        print(f"Documents : {len(docs)}")
        print(f"Total size: {db.total_bytes()/1e6:.1f} MB")
        for doc in docs[:10]:
            print(f"  {doc.title[:50]:50s}  {doc.byte_len//1024:6d} KB")
        if len(docs) > 10:
            print(f"  ... and {len(docs)-10} more")

    db.close()
