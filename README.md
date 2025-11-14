# 🧠 Taller RAG con Solr y Milvus

Este proyecto implementa dos backends de recuperación para un pipeline tipo RAG:

- 🔹 **Solr** (búsqueda léxica con BM25)
- 🔹 **Milvus** (búsqueda vectorial con embeddings)

Ambos se exponen a través de una **API unificada en FastAPI**, que permite consultar:

- `/solr` → búsqueda léxica
- `/milvus` → búsqueda vectorial
- `/ask` → selector de backend (`solr`, `milvus` o ambos)

---

## 0. Requisitos

- 🐳 **Docker** y **Docker Compose**
- 🐍 **Python 3.10+** (para ejecutar scripts de conversión / indexación / evaluación)
- Archivo de datos:
  - `data/corpus/corpus_bloques_100.csv` (corpus original en CSV)

> ⚠️ Todo se asume desde la carpeta raíz del proyecto:  
> `C:\Users\User\rag-solr-milvus` (ajusta si tu ruta es distinta).

---

## 1. Estructura mínima del proyecto

```bash
rag-solr-milvus/
├── data/
│   └── corpus/
│       ├── corpus_bloques_100.csv      # Corpus original
│       └── corpus_texto.jsonl          # Corpus convertido (se genera)
├── services/
│   ├── api/
│   │   └── app.py                      # API FastAPI (Solr & Milvus)
│   ├── indexer/
│   │   ├── convertir_csv.py
│   │   ├── indexar_solr.py
│   │   └── index_milvus.py
│   ├── solr/                           # Configuración del core rag2
│   └── milvus/                         # Notas de la colección
├── reports/                            # Métricas y gráficos (se generan)
├── docker-compose.yml
└── README.md
2. Levantar los servicios (Solr, Milvus, API)
Desde la raíz del proyecto:

bash
Copiar código
docker compose up -d --build
Esto levanta:

solr → en http://localhost:8983

milvus → puerto 19530

api (FastAPI) → http://localhost:8000

etcd, minio → dependencias de Milvus

solr-init → inicializa el core rag2

Verificar:

bash
Copiar código
docker compose ps
3. Preparar el corpus (CSV → JSONL)
Solo si aún no existe data/corpus/corpus_texto.jsonl.

3.1. Crear entorno (opcional, pero recomendado)
bash
Copiar código
python -m venv .venv
.venv\Scripts\activate
3.2. Instalar dependencias de los scripts
bash
Copiar código
pip install -r services/indexer/requirements.txt
3.3. Convertir el CSV a JSONL
bash
Copiar código
python services/indexer/convertir_csv.py ^
  --input data/corpus/corpus_bloques_100.csv ^
  --output data/corpus/corpus_texto.jsonl ^
  --text-col texto_limpio
texto_limpio es el nombre de la columna que contiene el texto en el CSV.

4. Indexar en Solr y Milvus
4.1. Indexar en Solr (BM25)
bash
Copiar código
python services/indexer/indexar_solr.py
Este script:

Hace ping a http://localhost:8983/solr/rag2

Lee data/corpus/corpus_texto.jsonl

Inserta cada documento en el core rag2 con campos:

id

text

Si todo sale bien verás algo como:

text
Copiar código
Ping: {'status': 'OK', ...}
Indexed N documents into Solr core 'rag2'
4.2. Indexar en Milvus (vectorial)
bash
Copiar código
python services/indexer/index_milvus.py ^
  --input data/corpus/corpus_texto.jsonl ^
  --host 127.0.0.1 ^
  --port 19530
Este script:

Conecta a Milvus en 127.0.0.1:19530

Crea la colección corpus_rag (si no existe)

Usa el modelo paraphrase-multilingual-MiniLM-L12-v2 para generar embeddings

Inserta los registros con campos:

id

text

embedding

Salida típica:

text
Copiar código
Indexando: 100%|██████████████| 12/12 [00:15]
OK -> 725 chunks en colección 'corpus_rag'
5. Probar la API paso a paso
5.1. Comprobar que la API está viva
bash
Copiar código
curl http://localhost:8000/health
Respuesta esperada:

json
Copiar código
{"status":"ok"}
5.2. Probar endpoint Solr (/solr)
Ejemplo:

bash
Copiar código
curl "http://localhost:8000/solr?q=paz territorial&k=5"
Respuesta esperada (lista de documentos):

json
Copiar código
[
  {
    "source": "solr",
    "id": "doc_000000",
    "text": "presentación del informe final Hay futuro si hay verdad...",
    "score": 7.23
  },
  ...
]
5.3. Probar endpoint Milvus (/milvus)
bash
Copiar código
curl "http://localhost:8000/milvus?q=paz territorial&k=5"
Respuesta similar, con "source": "milvus".

5.4. Probar endpoint unificado /ask
bash
Copiar código
curl "http://localhost:8000/ask?query=paz territorial&backend=solr&k=5"
curl "http://localhost:8000/ask?query=paz territorial&backend=milvus&k=5"
curl "http://localhost:8000/ask?query=paz territorial&backend=both&k=5"
backend=solr → solo Solr

backend=milvus → solo Milvus

backend=both → concatena resultados de ambos backends

5.5. UI interactiva (Swagger / Redoc)
Abrir en el navegador:

Swagger: 👉 http://localhost:8000/docs

Redoc: 👉 http://localhost:8000/redoc

Desde ahí puedes probar /solr, /milvus y /ask con formularios.

6. Evaluar el desempeño (opcional, pero recomendado)
Si quieres medir métricas tipo recall, MRR, nDCG, etc., usa el evaluador.

6.1. Archivo de queries + gold
El evaluador usa:

data/queries_gold.jsonl
con campos:

query → texto de la pregunta

gold_ids → lista de IDs relevantes (por ejemplo ["doc_000000"])

6.2. Ejecutar el evaluador
bash
Copiar código
python services/evaluator/evaluator.py
Esto:

Evalúa primero Solr, luego Milvus

Llama a la API /ask con backend=solr y backend=milvus

Calcula métricas por query y resumen agregados

Genera:

reports/metrics_per_query.csv

reports/metrics_summary.csv

Gráficos:

reports/latency_comparison.png

reports/recall_comparison.png

reports/mrr_comparison.png

reports/ndcg_comparison.png

reports/rougeL_comparison.png

Ejemplo de resumen (contenido típico):

text
Copiar código
Resumen:
           latency  recall_at_k       mrr      ndcg    rougeL
backend
milvus   0.263548          1.0  0.939394  0.940064  0.161310
solr     0.136780          1.0  0.893939  0.920994  0.002627
7. Reiniciar todo (si algo se rompe)
Si cambiaste app.py, el schema de Solr, o tuviste algún problema:

bash
Copiar código
docker compose down
docker compose up -d --build
Luego repetir:

Conversión (si hace falta)

indexar_solr.py

index_milvus.py

8. Accesos rápidos
🧠 API FastAPI: http://localhost:8000

📚 Docs Swagger: http://localhost:8000/docs

🔎 Solr UI: http://localhost:8983/solr/#/rag2/query

🧮 Milvus: conecta vía Python usando pymilvus y host=localhost, port=19530
