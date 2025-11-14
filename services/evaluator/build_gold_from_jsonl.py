import json
from pathlib import Path

# Rutas base
ROOT = Path(__file__).resolve().parents[2]  # .../rag-solr-milvus
JSONL_IN = ROOT / "data" / "corpus" / "corpus_texto.jsonl"
QUERIES_OUT = ROOT / "data" / "queries_gold.jsonl"

# Cuántas palabras usamos para la query y para el gold_answer (para ROUGE-L)
N_TOKENS_QUERY = 20
N_TOKENS_GOLD_ANSWER = 60

def shorten_text(text: str, n_tokens: int) -> str:
    tokens = str(text).strip().split()
    if len(tokens) <= n_tokens:
        return " ".join(tokens)
    return " ".join(tokens[:n_tokens])

def main():
    if not JSONL_IN.exists():
        raise FileNotFoundError(f"No encuentro el JSONL de corpus: {JSONL_IN}")

    QUERIES_OUT.parent.mkdir(parents=True, exist_ok=True)

    with JSONL_IN.open("r", encoding="utf-8") as f_in, \
         QUERIES_OUT.open("w", encoding="utf-8") as f_out:

        for i, line in enumerate(f_in):
            obj = json.loads(line)
            doc_id = str(obj.get("id"))          # ej. "doc_000123"
            text   = str(obj.get("text", ""))    # texto completo

            if not text.strip():
                continue

            query = shorten_text(text, N_TOKENS_QUERY)
            gold_answer = shorten_text(text, N_TOKENS_GOLD_ANSWER)

            rec = {
                "id": i + 1,              # id interno de query
                "query": query,           # lo que se enviará a /ask
                "gold_ids": [doc_id],     # este es el id que debe recuperar Solr/Milvus
                "gold_answer": gold_answer
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Gold standard generado en: {QUERIES_OUT}")

if __name__ == "__main__":
    main()
