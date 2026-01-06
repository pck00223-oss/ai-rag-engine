# python/rag_cli.py
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import List, Dict

from retrieval_bm25 import BM25Index, tokenize
from db import DB_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_EXE = PROJECT_ROOT / "build" / "Release" / "llm_cli.exe"
MODEL = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct-q5_k_m.gguf"

TOPK = 5

# =========================
# 阀门（偏宽松，但保证不乱答）
# =========================
MIN_BM25_SCORE = 0.6
MIN_COVERAGE = 0.10

# query 里出现这些“硬术语”，证据中必须真的出现，否则不能答
HARD_TERMS = {
    "bm25", "faiss", "rerank", "embedding", "vector",
    "flask", "python",
    "std::format", "format", "printf",
}

_CITE_RE = re.compile(r"\[chunk:\d+\]")

def extract_model_output(text: str) -> str:
    if not text:
        return ""
    a = text.find("--- model output ---")
    b = text.find("--- end ---")
    if a != -1 and b != -1 and b > a:
        return text[a + len("--- model output ---"):b].strip()
    return text.strip()

def call_llm(prompt: str) -> str:
    if not LLM_EXE.exists():
        raise RuntimeError(f"找不到 llm_cli.exe: {LLM_EXE}")
    if not MODEL.exists():
        raise RuntimeError(f"找不到模型: {MODEL}")

    cmd = [str(LLM_EXE), "-m", str(MODEL), "-p", prompt]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    raw = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
    return extract_model_output(raw)

def save_run(query: str, topk: int, chunk_ids: List[int], prompt: str, answer: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO runs(query, topk, chunk_ids_json, prompt, answer) VALUES(?,?,?,?,?)",
        (query, topk, json.dumps(chunk_ids, ensure_ascii=False), prompt, answer),
    )
    conn.commit()
    conn.close()

def normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mx = max(scores)
    if mx <= 0:
        return [0.0 for _ in scores]
    return [s / mx for s in scores]

def token_coverage(q_tokens: List[str], text: str) -> float:
    if not q_tokens:
        return 0.0
    t = (text or "").lower()
    hit, total = 0, 0
    for tok in q_tokens:
        tok = (tok or "").strip().lower()
        if not tok:
            continue
        # 过滤中文单字（太噪）
        if len(tok) == 1 and re.fullmatch(r"[\u4e00-\u9fff]", tok):
            continue
        total += 1
        if tok in t:
            hit += 1
    return (hit / total) if total else 0.0

def query_hard_terms(q: str) -> List[str]:
    ql = (q or "").lower()
    return [t for t in HARD_TERMS if t in ql]

def evidence_has_terms(hits: List[Dict], terms: List[str]) -> bool:
    if not terms:
        return True
    blob = "\n".join([(h.get("title","") + "\n" + h.get("text","")).lower() for h in hits])
    return all(t in blob for t in terms)  # 注意：这里用 all，更严格（问 flask 就得真有 flask）

def title_hit(q_tokens: List[str], title: str) -> bool:
    tl = (title or "").lower()
    return any((t.lower() in tl) for t in q_tokens if len(t) >= 2)

def build_prompt(query: str, hits: List[Dict]) -> str:
    cites = []
    for h in hits:
        cites.append(f"[chunk:{h['chunk_id']}] {h['title']}\n{h['text']}\n")
    context = "\n".join(cites)

    return (
        "你是一个检索增强问答助手（RAG）。你【只能】基于下面【证据】回答。\n"
        "硬性规则：\n"
        "1) 回答中【必须】包含至少一个引用，格式如 [chunk:226]。\n"
        "2) 如果证据不足，直接输出：证据不足（不要解释）。\n"
        "3) 不要编造。\n\n"
        f"【问题】\n{query}\n\n"
        f"【证据】\n{context}\n\n"
        "【回答】\n"
    )

def pick_best_excerpt_hit(query: str, hits: List[Dict]) -> Dict | None:
    """
    关键修复：fallback 不再盲选 hits[0]，
    而是优先选“包含硬术语 / 覆盖率高 / 标题命中”的那个。
    """
    if not hits:
        return None

    qtok = tokenize(query)
    hard = query_hard_terms(query)

    def score(h: Dict) -> float:
        text = (h.get("text") or "").lower()
        title = (h.get("title") or "").lower()
        hard_ok = 1.0 if (not hard or all(t in (title + "\n" + text) for t in hard)) else 0.0
        cov = float(h.get("cov", 0.0))
        th = 1.0 if title_hit(qtok, title) else 0.0
        bm = float(h.get("score", 0.0))
        # hard_ok 权重最大，避免摘到无关 chunk
        return 10.0 * hard_ok + 2.0 * th + 3.0 * cov + 0.05 * bm

    return max(hits, key=score)

def fallback_excerpt_answer(query: str, hits: List[Dict]) -> str:
    best = pick_best_excerpt_hit(query, hits)
    if not best:
        return "证据不足"
    cid = best["chunk_id"]
    title = best.get("title", "")
    text = (best.get("text", "") or "").strip()
    snippet = text[:900].rstrip()

    return (
        "结论：根据知识库摘录如下（模型未按引用格式输出，已自动改为摘录回答）。\n"
        f"- 来源：[chunk:{cid}] {title}\n"
        "摘录：\n"
        f"{snippet}"
    )

def main():
    bm25 = BM25Index()

    while True:
        q = input("Query> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        hits = bm25.search(q, TOPK)
        qtok = tokenize(q)

        bm25_scores = [float(h.get("score", 0.0)) for h in hits]
        bm25_norm = normalize(bm25_scores)

        enriched: List[Dict] = []
        for i, h in enumerate(hits):
            cov = token_coverage(qtok, h.get("text", ""))
            th = 1.0 if title_hit(qtok, h.get("title", "")) else 0.0
            # final：BM25 为主，覆盖率辅助，标题命中加一点
            final = 0.75 * (bm25_norm[i] if i < len(bm25_norm) else 0.0) + 0.25 * cov + 0.08 * th

            hh = dict(h)
            hh["cov"] = cov
            hh["final"] = final
            enriched.append(hh)

        enriched.sort(key=lambda x: x.get("final", 0.0), reverse=True)

        print("\n=== HITS ===")
        for h in enriched:
            print(
                f"final={h.get('final',0.0):.3f} "
                f"bm25={h.get('score',0.0):.3f} "
                f"cov={h.get('cov',0.0):.3f} "
                f"chunk={h.get('chunk_id')} "
                f"title={h.get('title','')}"
            )

        # 1) 强边界：硬术语必须在证据中出现，否则直接证据不足
        hard = query_hard_terms(q)
        if hard and not evidence_has_terms(enriched, hard):
            answer = "证据不足"
            save_run(q, TOPK, [h["chunk_id"] for h in enriched], "(no_prompt)", answer)
            print("\n=== ANSWER ===")
            print(answer)
            print("\n(saved to runs)\n")
            continue

        # 2) 过滤：BM25 或 coverage 达标即可（避免太严）
        filtered = [
            h for h in enriched
            if float(h.get("score", 0.0)) >= MIN_BM25_SCORE
            or float(h.get("cov", 0.0)) >= MIN_COVERAGE
            or title_hit(qtok, h.get("title", ""))
        ]

        if not filtered:
            answer = "证据不足"
            save_run(q, TOPK, [], "(no_prompt)", answer)
            print("\n=== ANSWER ===")
            print(answer)
            print("\n(saved to runs)\n")
            continue

        # 3) 调模型
        prompt = build_prompt(q, filtered)
        answer = call_llm(prompt)

        # 4) 必须带引用；否则用“选对 chunk 的摘录 fallback”
        reason = "ok"
        if not _CITE_RE.search(answer or ""):
            answer = fallback_excerpt_answer(q, filtered)
            reason = "fallback_excerpt"

        chunk_ids = [h["chunk_id"] for h in filtered]
        save_run(q, TOPK, chunk_ids, prompt, answer)

        print("\n=== ANSWER ===")
        print(answer)
        print(f"\n(saved to runs, reason={reason})\n")

if __name__ == "__main__":
    main()
