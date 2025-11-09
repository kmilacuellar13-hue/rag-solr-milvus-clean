# Milvus (colección `corpus_rag`)

Campos:
- `id` (VARCHAR, PK, max_length=64)
- `embedding` (FLOAT_VECTOR, dim=384, métrica COSINE, AUTOINDEX)
- `text` (VARCHAR, max_length=8192)

Creación/carga se hace desde `services/indexer/index_milvus.py`.
El servicio expone:
- gRPC / SDK: 19530
- REST monitor: 9091



