from flask import Flask, request, jsonify
import numpy as np

try:
    import faiss  # pip install faiss-cpu
except Exception:
    faiss = None

app = Flask(__name__)

DIM = 128
TOPK_DEFAULT = 5
TOPK_MAX = 20


def create_faiss_index(dimension: int):
    if faiss is None:
        raise RuntimeError("faiss 未安装。请先 pip install faiss-cpu")
    return faiss.IndexFlatL2(dimension)


def add_to_faiss(index, docs: np.ndarray):
    docs = np.asarray(docs)
    if docs.dtype != np.float32:
        docs = docs.astype(np.float32)
    if docs.ndim != 2 or docs.shape[1] != DIM:
        raise ValueError(f"docs must be shape (N, {DIM}), got {docs.shape}")
    index.add(docs)
    return index


def faiss_search(index, query: np.ndarray, k: int):
    # query: shape (1, DIM), float32
    distances, indices = index.search(query, k)
    return indices[0], distances[0]


# ---- init demo index ----
index = None
if faiss is not None:
    index = create_faiss_index(DIM)
    docs = np.random.random((10, DIM)).astype(np.float32)
    add_to_faiss(index, docs)


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        ok=True,
        message='Server is running. Use POST /search with JSON {"query": [128 floats]}',
        dim=DIM,
        topk_default=TOPK_DEFAULT,
        faiss_installed=(faiss is not None),
        index_ready=(index is not None),
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(ok=True)


@app.route("/docs", methods=["GET"])
def docs():
    return jsonify(
        endpoints={
            "GET /": "basic info",
            "GET /health": "health check",
            "GET /docs": "this help",
            "POST /search": 'JSON {"query":[128 floats], "topk": optional int<=20}',
        },
        curl_example=[
            'python -c "import json; print(json.dumps({\'query\':[0.0]*128}))" > payload.json',
            'curl.exe -X POST "http://127.0.0.1:5000/search" -H "Content-Type: application/json" --data-binary "@payload.json"',
        ],
    )


@app.route("/search", methods=["POST"])
def search():
    if index is None:
        return jsonify(error="FAISS index not initialized. Is faiss installed?"), 500

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Invalid JSON body. Expect a JSON object."), 400

    query = data.get("query", None)
    if not isinstance(query, list):
        return jsonify(error=f"Field 'query' must be a list of length {DIM}."), 400
    if len(query) != DIM:
        return jsonify(error=f"Query vector must be of dimension {DIM}, got {len(query)}"), 400

    # 可选 topk
    topk = data.get("topk", TOPK_DEFAULT)
    if not isinstance(topk, int) or topk <= 0:
        return jsonify(error="Field 'topk' must be a positive integer."), 400
    if topk > TOPK_MAX:
        return jsonify(error=f"topk too large. Max is {TOPK_MAX}."), 400

    try:
        q = np.array(query, dtype=np.float32).reshape(1, -1)
    except Exception:
        return jsonify(error="Query contains non-numeric values."), 400

    idxs, dists = faiss_search(index, q, topk)
    return jsonify(indices=idxs.tolist(), distances=dists.tolist())


# 如果有人用浏览器 GET /search，给出明确提示
@app.route("/search", methods=["GET"])
def search_get_not_allowed():
    return jsonify(error="Use POST /search with JSON body. See /docs"), 405


if __name__ == "__main__":
    # 本机调试：127.0.0.1
    # 局域网访问：改成 0.0.0.0，然后用你电脑的 IPv4 访问（不要用 0.0.0.0）
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
