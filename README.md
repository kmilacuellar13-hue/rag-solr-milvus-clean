# Taller RAG con Solr y Milvus — Entrega funcional

Este repositorio implementa dos pipelines RAG: léxico con **Apache Solr** y vectorial con **Milvus**, expuestos por una **API unificada (FastAPI)**. Incluye scripts de conversión/indexación y un evaluador con métricas.

## Estructura
- `/data/corpus/`: CSV original y JSONL generado.
- `/services/api/`: API FastAPI (Solr & Milvus).
- `/services/indexer/`: convertir CSV → JSONL, indexar en Solr/Milvus, evaluar.
- `/services/solr/`: schema y script opcional de inicialización.
- `/services/milvus/`: notas de colección.
- `/reports/`: resultados del evaluador.

## Requisitos
- Docker y Docker Compose.
- Python 3.10+ (para ejecutar los indexadores/evaluador desde host).
- `data/corpus/corpus_bloques_100.csv`.

## Pasos
1. **Levantar servicios**
docker compose up -d --build


2. Preparar e indexar
pip install -r services/indexer/requirements.txt

python services/indexer/convertir_csv.py `
  --input data/corpus/corpus_bloques_100.csv `
  --output data/corpus/corpus_texto.jsonl `
  --text-col texto_limpio

python services/indexer/indexar_solr.py `
  --solr http://localhost:8983/solr/rag2 `
  --input data/corpus/corpus_texto.jsonl

--solr http://localhost:8983/solr/rag2

python services/indexer/index_milvus.py --host localhost --port 19530

3. Probar API

curl "http://localhost:8000/ask?query=paz%20territorial&backend=solr&k=3"
curl "http://localhost:8000/ask?query=paz%20territorial&backend=milvus&k=3"

4.Evaluar

# Evaluar Solr
python services/indexer/evaluator.py \
  --backend solr \
  --queries data/corpus/queries.jsonl \
  --gold data/corpus/gold.jsonl \
  --k 5

Solr: http://localhost:8983

API: http://localhost:8000/docs

# Evaluar Milvus
python services/indexer/evaluator.py \
  --backend milvus \
  --queries data/corpus/queries.jsonl \
  --gold data/corpus/gold.jsonl \
  --k 5

Resultados en /reports.

Notas
Solr usa el campo text (schema en /services/solr/schema.json).
Milvus usa colección corpus_rag con campos: id (PK), embedding (FLOAT_VECTOR dim=384), text.



