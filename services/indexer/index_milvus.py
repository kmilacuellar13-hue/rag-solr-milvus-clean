import argparse
import json
from pathlib import Path

from pymilvus import (
    connections, utility, Collection, FieldSchema, CollectionSchema, DataType
)
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# ==== Config ====
COLLECTION_NAME = "corpus_rag"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384 dims
CHUNK_SIZE = 4000                      # tamaño seguro (4k chars)
CHUNK_OVERLAP = 200                    # solape para coherencia
MAX_VARCHAR = 65535                    # límite Milvus
BATCH = 64

def chunk_text(txt, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not txt:
        return []
    txt = str(txt)
    if len(txt) <= size:
        return [txt]
    chunks = []
    i = 0
    while i < len(txt):
        end = min(i + size, len(txt))
        chunks.append(txt[i:end])
        if end == len(txt):
            break
        i = end - overlap
        if i < 0:
            i = 0
    return chunks

def ensure_collection(dim: int):
    # Elimina colección previa si existe (idempotente para demo)
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="RAG corpus chunks")
    coll = Collection(name=COLLECTION_NAME, schema=schema)
    # Index sobre vector
    coll.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}},
    )
    coll.load()
    return coll

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/corpus/corpus_texto.jsonl", help="JSONL con {'id','text'} o {'id','texto_limpio'}")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="19530")
    args = ap.parse_args()

    connections.connect(alias="default", host=args.host, port=args.port)

    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    coll = ensure_collection(dim)

    # Cargar JSONL
    path = Path(args.input)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pid = str(obj.get("id", ""))
            txt = obj.get("text") or obj.get("texto_limpio") or ""
            rows.append((pid, txt))

    # Expandir a chunks
    expanded = []
    for pid, txt in rows:
        for j, ch in enumerate(chunk_text(txt)):
            # última defensa: recorta si supera VARCHAR (muy raro con CHUNK_SIZE=4000)
            if len(ch) > MAX_VARCHAR:
                ch = ch[:MAX_VARCHAR]
            expanded.append((pid, j, ch))

    # Embeddings por lotes
    ids, chunk_ids, texts, vecs = [], [], [], []
    for i in tqdm(range(0, len(expanded), BATCH), desc="Indexando"):
        batch = expanded[i:i+BATCH]
        bt_texts = [b[2] for b in batch]
        embs = model.encode(bt_texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        for (pid, j, ch), v in zip(batch, embs):
            ids.append(pid)
            chunk_ids.append(j)
            texts.append(ch)
            vecs.append(v.astype(np.float32))

        if len(vecs) >= 512:  # flush intermedio
            coll.insert([ids, chunk_ids, texts, vecs])  # auto_id para PK
            ids, chunk_ids, texts, vecs = [], [], [], []

    if vecs:
        coll.insert([ids, chunk_ids, texts, vecs])

    coll.flush()
    print(f"OK -> {len(expanded)} chunks en colección '{COLLECTION_NAME}'")

if __name__ == "__main__":
    main()
