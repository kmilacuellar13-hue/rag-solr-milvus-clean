from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, requests

# =========================
# Config
# =========================
SOLR_URL = os.environ.get("SOLR_URL", "http://solr:8983/solr/rag2")
MILVUS_HOST = os.environ.get("MILVUS_HOST", "milvus")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "corpus_rag")

app = FastAPI(title="RAG Solr+Milvus API", version="1.1.0")

# CORS para UI local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class SearchResponse(BaseModel):
    source: str
    id: str | None = None
    text: str | None = None
    score: float | None = None

@app.get("/")
def root():
    return {"service": "RAG Solr+Milvus API", "ok": True}

# =========================
# Health
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# SOLR (BM25 / keyword)
# =========================
@app.get("/solr")
def solr_query(q: str = Query(..., min_length=1), k: int = 5):
    try:
        r = requests.get(
            f"{SOLR_URL}/select",
            params={"q": f"text:{q}", "rows": k, "fl": "id,text", "wt": "json"},
            timeout=15
        )
        r.raise_for_status()
        docs = r.json().get("response", {}).get("docs", [])
        out: list[SearchResponse] = []
        for d in docs:
            txt = d.get("text", "")
            if isinstance(txt, list):
                txt = txt[0]
            out.append(SearchResponse(source="solr", id=d.get("id"), text=txt))
        return out
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Solr error: {e}")

# =========================
# MILVUS (Vector search)
# =========================
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

_MODEL = None
# Usa el MISMO modelo que indexaste
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL

def milvus_connect():
    # Conexión "default" idempotente
    if not connections.has_connection("default"):
        connections.connect("default", host=MILVUS_HOST, port=str(MILVUS_PORT))

@app.get("/milvus")
def milvus_search(q: str = Query(..., min_length=1), k: int = 5):
    try:
        milvus_connect()
        model = get_model()
        # Normaliza embeddings (igual que en indexación)
        emb = model.encode([q], normalize_embeddings=True).tolist()

        col = Collection(MILVUS_COLLECTION)
        try:
            col.load()  # idempotente en 2.4.x
        except Exception:
            pass

        # Si tu índice es IVF usa nprobe; si es HNSW ajusta a "ef"
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        res = col.search(
            data=emb,
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["text"]
        )

        out: list[SearchResponse] = []
        for hit in res[0]:
            txt = hit.entity.get("text")
            if isinstance(txt, list):
                txt = txt[0]
            out.append(SearchResponse(
                source="milvus",
                id=str(hit.id),
                text=txt or "",
                score=float(hit.distance)  # en 2.4.x: .distance
            ))
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus error: {e}")

# =========================
# /ask -> unifica ambos
# =========================
@app.get("/ask")
def ask(
    query: str = Query(..., min_length=1),
    backend: str = Query("both", pattern="^(solr|milvus|both)$"),
    k: int = 5
):
    results: list[SearchResponse] = []
    if backend in ("solr", "both"):
        results += solr_query(query, k)
    if backend in ("milvus", "both"):
        results += milvus_search(query, k)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
