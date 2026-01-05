import argparse
import os
import sqlite3
import subprocess
from typing import List, Tuple

from BM25 import BM25Index, search as bm25_search


def sqlite_fetch_by_ids(db_path: str, ids: List[int]) -> List[Tuple]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    q = ",".join(["?"] * len(ids))
    cur.execute(
        f"""
        SELECT id, doc_id, doc_path, chunk_id, text, line_start, line_end
        FROM documents
        WHERE id IN ({q})
        """,
        ids,
    )
    rows = cur.fetchall()
    conn.close()

    # 保持与 ids 顺序一致
    mp = {int(r[0]): r for r in rows}
    return [mp[i] for i in ids if i in mp]


def build_context_with_citations(rows: List[Tuple]) -> str:
    """
    rows: (id, doc_id, doc_path, chunk_id, text, line_start, line_end)
    """
    blocks = []
    for r in rows:
        _id = int(r[0])
        doc_id = r[1]
        chunk_id = int(r[3])
        text = (r[4] or "").strip()
        ls = r[5] if r[5] is not None else "?"
        le = r[6] if r[6] is not None else "?"
        cite = f"[{doc_id}#chunk{chunk_id}#L{ls}-L{le}]"
        blocks.append(f"{cite}\n{text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, evidence: str) -> str:
    # 这里用“必须引用”的硬约束，保证 M1 输出带证据
    return (
        "你是检索增强问答系统，只能使用给定证据回答。\n"
        "规则：\n"
        "1) 回答必须引用证据，引用格式必须是 [doc#chunk#Lx-Ly]。\n"
        "2) 如果证据不足以回答，直接回答：无法从证据中确定，并说明缺少什么。\n"
        "3) 不要编造。\n\n"
        "【证据】\n"
        f"{evidence}\n\n"
        "【问题】\n"
        f"{question}\n\n"
        "【回答】"
    )


def run_llm_cli(llm_exe: str, model_path: str, prompt: str) -> str:
    # 通过 --prompt 传入（注意 Windows 引号由 subprocess 处理）
    cmd = [
        llm_exe,
        "--model", model_path,
        "--prompt", prompt,
        "--n", "256",
        "--temp", "0.2",
        "--topk", "40",
        "--topp", "0.9",
        "--seed", "42",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    # 你的 llm_cli 会打印 --- model output --- ... --- end ---
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--db", default="data/documents.db")
    ap.add_argument("--bm25", default="data/bm25.pkl")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--llm_exe", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    if not os.path.exists(args.bm25):
        raise FileNotFoundError(f"BM25 index not found: {args.bm25}, run ingest.py first")
    if not os.path.exists(args.db):
        raise FileNotFoundError(f"SQLite db not found: {args.db}, run ingest.py first")

    idx = BM25Index.load(args.bm25)
    hits = bm25_search(idx, args.question, top_k=args.topk)
    hit_ids = [h[0] for h in hits]

    if not hit_ids:
        evidence = ""
        prompt = build_prompt(args.question, evidence)
        print(run_llm_cli(args.llm_exe, args.model, prompt))
        return

    rows = sqlite_fetch_by_ids(args.db, hit_ids)
    evidence = build_context_with_citations(rows)

    print("=== RETRIEVAL (TOPK) ===")
    for (rid, score), row in zip(hits, rows):
        doc_id = row[1]
        chunk_id = row[3]
        ls = row[5]
        le = row[6]
        print(f"- score={score:.4f}  [{doc_id}#chunk{chunk_id}#L{ls}-L{le}]")

    prompt = build_prompt(args.question, evidence)
    print("\n=== LLM OUTPUT ===")
    print(run_llm_cli(args.llm_exe, args.model, prompt))


if __name__ == "__main__":
    main()
