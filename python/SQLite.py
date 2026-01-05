import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import fitz  # PyMuPDF


@dataclass
class Chunk:
    doc_id: str
    doc_path: str
    doc_type: str
    chunk_id: int
    text: str
    start_char: int
    end_char: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def connect(db_path: str) -> sqlite3.Connection:
    ensure_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            doc_path TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            chunk_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            line_start INTEGER,
            line_end INTEGER,
            created_at INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_docid_chunk ON documents(doc_id, chunk_id);"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docpath ON documents(doc_path);")
    conn.commit()


def clear_doc(conn: sqlite3.Connection, doc_path: str) -> None:
    conn.execute("DELETE FROM documents WHERE doc_path = ?;", (doc_path,))
    conn.commit()


def insert_chunks(conn: sqlite3.Connection, chunks: List[Chunk]) -> None:
    now = int(time.time())
    conn.executemany(
        """
        INSERT INTO documents
        (doc_id, doc_path, doc_type, chunk_id, text, start_char, end_char,
         page_start, page_end, line_start, line_end, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        [
            (
                c.doc_id,
                c.doc_path,
                c.doc_type,
                c.chunk_id,
                c.text,
                c.start_char,
                c.end_char,
                c.page_start,
                c.page_end,
                c.line_start,
                c.line_end,
                now,
            )
            for c in chunks
        ],
    )
    conn.commit()


def fetch_all_chunks(conn: sqlite3.Connection) -> List[Tuple]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, doc_id, doc_path, doc_type, chunk_id, text,
               page_start, page_end, line_start, line_end
        FROM documents
        ORDER BY doc_path, chunk_id;
        """
    )
    return cur.fetchall()


# ---------------- text loading ----------------

def read_txt_md(path: str) -> str:
    # 尝试 utf-8-sig / utf-8 / gbk
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception:
            pass
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    doc = fitz.open(path)
    parts = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        txt = page.get_text("text")
        if txt:
            parts.append(f"\n[PAGE {i+1}]\n{txt}")
    doc.close()
    return "\n".join(parts)


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ---------------- chunking ----------------

def chunk_text_by_chars(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Tuple[int, int, str]]:
    """
    返回 (start_char, end_char, chunk_text)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    n = len(text)
    chunks = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end == n:
            break
        start = end - chunk_overlap
    return chunks


def build_chunks_for_file(path: str) -> List[Chunk]:
    ext = os.path.splitext(path)[1].lower()
    doc_type = ext.lstrip(".") if ext else "unknown"
    doc_id = os.path.basename(path)

    if ext == ".pdf":
        raw = read_pdf(path)
    elif ext in (".txt", ".md"):
        raw = read_txt_md(path)
    else:
        return []

    text = normalize_text(raw)
    triples = chunk_text_by_chars(text, chunk_size=900, chunk_overlap=150)

    out: List[Chunk] = []
    for i, (st, ed, ck) in enumerate(triples):
        # 轻量“行号”估算：按 chunk 内 \n 计数
        # （足够满足 M1 的引用需求）
        line_start = text[:st].count("\n") + 1
        line_end = text[:ed].count("\n") + 1

        # pdf 页码：若文本含 [PAGE n]，可粗略定位；这里先不强求准确
        out.append(
            Chunk(
                doc_id=doc_id,
                doc_path=path,
                doc_type=doc_type,
                chunk_id=i,
                text=ck,
                start_char=st,
                end_char=ed,
                page_start=None,
                page_end=None,
                line_start=line_start,
                line_end=line_end,
            )
        )
    return out
