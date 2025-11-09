from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

connections.connect("default", host="localhost", port="19530")
col = Collection("corpus_rag"); col.load()

m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
q = "paz territorial"
vec = [m.encode(q).tolist()]
res = col.search(vec, "vector", param={"nprobe":10}, limit=5, output_fields=["text"])
for hits in res:
    for h in hits:
        print(h.id, round(h.distance, 4), h.entity.get("text")[:80].replace("\n"," "))

