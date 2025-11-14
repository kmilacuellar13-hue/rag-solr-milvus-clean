🧠 Taller RAG con Solr y Milvus — Entrega Final

Este repositorio implementa dos pipelines de Recuperación Aumentada por Generación (RAG):

RAG léxico usando Apache Solr (BM25)

RAG vectorial usando Milvus (embeddings)

Ambos motores se exponen mediante una API unificada en FastAPI, y el sistema incluye scripts automáticos para:

Conversión del corpus

Indexación en Solr y Milvus

Evaluación del desempeño

Generación de métricas y gráficas comparativas

El proyecto cumple todos los criterios del taller RAG Solr–Milvus: indexación completa, API funcional, evaluador operativo y métricas obtenidas.

📁 Estructura del proyecto
rag-solr-milvus/
├── data/
│   └── corpus/
│       ├── corpus_bloques_100.csv      # Corpus original
│       └── corpus_texto.jsonl          # Corpus convertido (JSONL)
├── services/
│   ├── api/                            # API unificada (FastAPI)
│   ├── indexer/                        # Scripts de conversión, indexación y evaluación
│   ├── solr/                           # Configuración del core rag2
│   └── milvus/                         # Notas de la colección vectorial
├── reports/                            # Resultados generados por el evaluador
├── docker-compose.yml
└── README.md

⚙️ Requisitos

Docker + Docker Compose

Python 3.10+ (para ejecutar scripts de indexación/evaluación desde host)

Archivo de entrada:
data/corpus/corpus_bloques_100.csv

🚀 Ejecución del proyecto
1️⃣ Levantar el stack (Solr, Milvus, MinIO, ETCD, API)
docker compose up -d --build


Servicios incluidos:

Servicio	Rol
solr	Búsqueda BM25
milvus	Búsqueda vectorial
etcd	Coordinador de Milvus
minio	Almacenamiento de snapshots
api	API unificada FastAPI

Verificar:

docker compose ps

2️⃣ Conversión e indexación
Convertir CSV → JSONL
python services/indexer/convertir_csv.py ^
  --input data/corpus/corpus_bloques_100.csv ^
  --output data/corpus/corpus_texto.jsonl ^
  --text-col texto_limpio

Indexar en Solr
python services/indexer/indexar_solr.py ^
  --solr http://localhost:8983/solr/rag2 ^
  --input data/corpus/corpus_texto.jsonl

Indexar en Milvus
python services/indexer/index_milvus.py ^
  --input data/corpus/corpus_texto.jsonl ^
  --host localhost ^
  --port 19530

🔍 3️⃣ Probar la API
Salud general
curl http://localhost:8000/health

Consultar BM25 (Solr)
curl "http://localhost:8000/solr?q=paz territorial&k=5"

Consultar vectorial (Milvus)
curl "http://localhost:8000/milvus?q=paz territorial&k=5"

UI interactiva

👉 http://localhost:8000/docs

🧪 4️⃣ Evaluación (Solr vs Milvus)

El evaluador utiliza:

queries_gold.jsonl (queries reales)

corpus_texto.jsonl (documentos convertidos)

Ejecutar:
python services/evaluator/evaluator.py


Esto produce:

📄 reports/metrics_per_query.csv
📄 reports/metrics_summary.csv
📊 5 gráficos comparativos:

Latencia

Recall@k

MRR

nDCG

ROUGE-L

📊 Resultados de la evaluación (finales)

Con k = 5, ambos motores lograron recall perfecto.
Resultados globales:

Métrica	Milvus	Solr
Recall@k	1.00	1.00
MRR	0.939	0.894
nDCG	0.940	0.921
ROUGE-L	0.161	0.0026
Latencia (s)	0.263	0.136
📝 Conclusiones

Ambos motores alcanzan Recall@5 = 1.0, lo que demuestra que Solr y Milvus recuperan los documentos correctos dentro del top-k.

Milvus supera a Solr en métricas de ranking (MRR y nDCG), por lo que devuelve los documentos correctos en posiciones más altas del ranking.

Solr es más rápido, con latencias alrededor de 130 ms, lo cual es consistente con motores BM25 optimizados para matching léxico.

Milvus ofrece una calidad semántica notablemente superior, con ROUGE-L dos órdenes de magnitud mayor, evidenciando que los embeddings capturan mejor el contenido conceptual.

La combinación Solr + Milvus en una API unificada permite construir pipelines híbridos y escalables para tareas RAG.



