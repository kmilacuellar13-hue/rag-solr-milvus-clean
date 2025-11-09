FROM python:3.10-slim

WORKDIR /app

# Evita prompt interactivo
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/root/.cache/huggingface

# Dependencias nativas mínimas (wget/curl opcional para healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl && \
    rm -rf /var/lib/apt/lists/*

# Reqs e instalación
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# >>> PRE-DESCARGA DEL MODELO (mismo que usas en la API) <<<
# Cambia el nombre del modelo si tu app.py usa otro
ENV SBERT_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
import os
m = os.environ.get("SBERT_MODEL","sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("Downloading model:", m)
SentenceTransformer(m)
print("OK")
PY

# Copia de la app
COPY . /app

# Uvicorn
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]



