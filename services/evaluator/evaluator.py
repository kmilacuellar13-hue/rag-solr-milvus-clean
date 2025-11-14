import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from rouge_score import rouge_scorer

# =========================
# CONFIGURACIÓN GENERAL
# =========================

ROOT = Path(__file__).resolve().parents[2]  # .../rag-solr-milvus
API_URL = "http://localhost:8000/ask"

QUERIES_PATH = ROOT / "data" / "queries_gold.jsonl"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BACKENDS = ["solr", "milvus"]
K_DEFAULT = 5

# ROUGE-L
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# =========================
# MÉTRICAS
# =========================

def recall_at_k(gold_ids, retrieved_ids, k):
    if not gold_ids:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    inter = len(set(gold_ids) & set(retrieved_k))
    return inter / len(gold_ids)


def mrr(gold_ids, retrieved_ids):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0


def dcg(relevances):
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(gold_ids, retrieved_ids, k):
    rels = [1 if doc_id in gold_ids else 0 for doc_id in retrieved_ids[:k]]
    dcg_val = dcg(rels)
    ideal_rels = sorted(rels, reverse=True)
    idcg_val = dcg(ideal_rels) or 1.0
    return dcg_val / idcg_val


def rouge_l_score(gold_answer, predicted_answer):
    if not gold_answer or not predicted_answer:
        return 0.0
    scores = scorer.score(gold_answer, predicted_answer)
    return float(scores["rougeL"].fmeasure)


# =========================
# GOLD STANDARD
# =========================

def load_queries():
    """
    Lee queries_gold.jsonl con formato:
    {
      "id": 1,
      "query": "texto de query...",
      "gold_ids": ["doc_000000"],
      "gold_answer": "texto de referencia..."
    }
    """
    rows = []
    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# =========================
# LLAMADA A LA API /ask
# =========================

def call_api(query, backend, k=K_DEFAULT):
    """
    Llama a la API unificada:
    GET /ask?query=...&k=...&backend=solr|milvus|both

    La API devuelve:
    - una LISTA de SearchResponse:
        [
          {"source": "...", "id": "doc_000000", "text": "...", "score": ...},
          ...
        ]
    o, en versiones viejas:
        {"answer": "...", "sources": [ {...}, {...} ]}
    """
    params = {
        "query": query,
        "backend": backend,
        "k": int(k),
    }

    t0 = time.time()
    resp = requests.get(API_URL, params=params, timeout=120)
    latency = time.time() - t0

    resp.raise_for_status()
    data = resp.json()

    # Normalizar a lista de hits
    if isinstance(data, list):
        hits = data
    elif isinstance(data, dict) and "sources" in data:
        hits = data.get("sources", []) or []
    else:
        hits = []

    # "answer": usamos el texto del primer hit (no hay LLM acá)
    answer = ""
    if hits:
        first = hits[0]
        if isinstance(first, dict):
            answer = first.get("text") or first.get("answer", "") or ""

    # IDs recuperados (para métricas de ranking)
    retrieved_ids = [
        str(h.get("id"))
        for h in hits
        if isinstance(h, dict) and "id" in h
    ]

    return answer, retrieved_ids, latency



# =========================
# LLM-AS-A-JUDGE (GANCHO)
# =========================

def llm_as_judge_stub(query, answer_solr, answer_milvus):
    """
    Aquí podrías conectar un LLM externo (OpenAI, etc.)
    para comparar respuestas de Solr vs Milvus.

    Para el informe basta con describir el esquema
    y hacer 2-3 ejemplos manuales.
    """
    return None


# =========================
# EVALUACIÓN PRINCIPAL
# =========================

def main(k=K_DEFAULT):
    queries = load_queries()
    all_rows = []

    for backend in BACKENDS:
        print(f"\n=== Evaluando backend: {backend} ===")

        for row in queries:
            qid = row["id"]
            query = row["query"]
            gold_ids = [str(g) for g in row.get("gold_ids", [])]
            gold_answer = row.get("gold_answer", "")

            try:
                answer, retrieved_ids, latency = call_api(query, backend, k=k)
            except Exception as e:
                print(f"[ERROR] backend={backend}, qid={qid}: {e}")
                continue

            r_at_k = recall_at_k(gold_ids, retrieved_ids, k)
            mrr_val = mrr(gold_ids, retrieved_ids)
            ndcg_val = ndcg_at_k(gold_ids, retrieved_ids, k)
            rouge_val = rouge_l_score(gold_answer, answer)

            all_rows.append({
                "backend": backend,
                "query_id": qid,
                "query": query,
                "gold_ids": gold_ids,
                "latency": latency,
                "recall_at_k": r_at_k,
                "mrr": mrr_val,
                "ndcg": ndcg_val,
                "rougeL": rouge_val,
                "retrieved_ids": retrieved_ids,
                "answer": answer,
                "gold_answer": gold_answer,
            })

    df = pd.DataFrame(all_rows)
    per_query_path = REPORTS_DIR / "metrics_per_query.csv"
    df.to_csv(per_query_path, index=False, encoding="utf-8")
    print(f"\n✅ Métricas por query guardadas en {per_query_path}")

    if df.empty:
      print("⚠️ No se obtuvieron resultados (DataFrame vacío). Revisa que la API esté respondiendo bien.")
      return


    # Resumen por backend
    summary = df.groupby("backend")[["latency", "recall_at_k", "mrr", "ndcg", "rougeL"]].mean()
    summary_path = REPORTS_DIR / "metrics_summary.csv"
    summary.to_csv(summary_path)
    print(f"✅ Resumen por backend guardado en {summary_path}")
    print("\nResumen:\n", summary)

    # Gráficos comparativos
    plt.figure()
    summary["latency"].plot(kind="bar", title="Latencia promedio (s)")
    plt.ylabel("Segundos")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "latency_comparison.png")

    plt.figure()
    summary["recall_at_k"].plot(kind="bar", title=f"Recall@{k} promedio")
    plt.ylabel("Recall")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "recall_comparison.png")

    plt.figure()
    summary["mrr"].plot(kind="bar", title="MRR promedio")
    plt.ylabel("MRR")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "mrr_comparison.png")

    plt.figure()
    summary["ndcg"].plot(kind="bar", title=f"nDCG@{k} promedio")
    plt.ylabel("nDCG")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "ndcg_comparison.png")

    plt.figure()
    summary["rougeL"].plot(kind="bar", title="ROUGE-L promedio")
    plt.ylabel("F1 ROUGE-L")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "rougeL_comparison.png")

    print(f"✅ Gráficos guardados en {REPORTS_DIR}")


if __name__ == "__main__":
    main(k=5)
