import pickle
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize_zh_en(text: str) -> List[str]:
    """
    M1 简化分词：
    - 英文按单词
    - 中文按单字（足够做 BM25 baseline）
    """
    text = text.lower()
    # 提取英文词
    en = re.findall(r"[a-z0-9]+", text)
    # 提取中文字符
    zh = re.findall(r"[\u4e00-\u9fff]", text)
    return en + zh


@dataclass
class BM25Index:
    bm25: BM25Okapi
    doc_keys: List[int]     # 对应 SQLite 的 documents.id
    doc_meta: List[Tuple]   # (doc_id, doc_path, chunk_id, line_start, line_end)
    tokenized: List[List[str]]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_bm25_from_sql_rows(rows: List[Tuple]) -> BM25Index:
    """
    rows: (id, doc_id, doc_path, doc_type, chunk_id, text, page_start, page_end, line_start, line_end)
    """
    doc_keys = []
    doc_meta = []
    corpus_tokens = []
    for r in rows:
        doc_row_id = int(r[0])
        doc_id = r[1]
        doc_path = r[2]
        chunk_id = int(r[4])
        text = r[5] or ""
        line_start = r[8]
        line_end = r[9]

        toks = tokenize_zh_en(text)
        doc_keys.append(doc_row_id)
        doc_meta.append((doc_id, doc_path, chunk_id, line_start, line_end))
        corpus_tokens.append(toks)

    bm25 = BM25Okapi(corpus_tokens)
    return BM25Index(bm25=bm25, doc_keys=doc_keys, doc_meta=doc_meta, tokenized=corpus_tokens)


def search(index: BM25Index, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
    """
    return: [(sqlite_row_id, score)]
    """
    qtok = tokenize_zh_en(query)
    scores = index.bm25.get_scores(qtok)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return []

    top_idx = np.argsort(-scores)[:top_k]
    out = []
    for i in top_idx:
        out.append((index.doc_keys[int(i)], float(scores[int(i)])))
    return out
