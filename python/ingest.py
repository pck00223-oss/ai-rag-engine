import argparse
import os
from typing import List

from SQLite import connect, init_schema, clear_doc, insert_chunks, build_chunks_for_file, fetch_all_chunks
from BM25 import build_bm25_from_sql_rows


def list_docs(docs_dir: str) -> List[str]:
    exts = {".pdf", ".txt", ".md"}
    out = []
    for root, _, files in os.walk(docs_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                out.append(os.path.join(root, fn))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_dir", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--bm25", required=True)
    ap.add_argument("--rebuild", action="store_true", help="rebuild chunks for all docs")
    args = ap.parse_args()

    conn = connect(args.db)
    init_schema(conn)

    docs = list_docs(args.docs_dir)
    if not docs:
        print(f"[ingest] no docs found in {args.docs_dir}")
        return

    for p in docs:
        if args.rebuild:
            clear_doc(conn, p)
        chunks = build_chunks_for_file(p)
        if not chunks:
            print(f"[ingest] skip unsupported/empty: {p}")
            continue
        # 为简单起见：每次都先清空再插（避免重复）
        clear_doc(conn, p)
        insert_chunks(conn, chunks)
        print(f"[ingest] {os.path.basename(p)} -> chunks={len(chunks)}")

    rows = fetch_all_chunks(conn)
    bm25_index = build_bm25_from_sql_rows(rows)
    os.makedirs(os.path.dirname(args.bm25), exist_ok=True)
    bm25_index.save(args.bm25)
    print(f"[ingest] bm25 saved: {args.bm25}, total_chunks={len(rows)}")


if __name__ == "__main__":
    main()
