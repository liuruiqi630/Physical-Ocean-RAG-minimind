import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path=r"d:\minimind\Physical Oceanography RAG\chroma_db")
col = client.get_collection("wulihaiyang_notes")

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
query = "什么是地转流？它的平衡机制是什么？"
# BGE 查询侧加前缀
q_vec = model.encode(
    ["为这个句子生成表示以用于检索相关文章：" + query],
    normalize_embeddings=True,
).tolist()

res = col.query(
    query_embeddings=q_vec,
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

print("查询：", query)
print()
for i in range(3):
    print(f"===== Top {i + 1} =====")
    print("heading_path:", res["metadatas"][0][i]["heading_path"])
    print("cosine distance:", round(res["distances"][0][i], 4))
    print(res["documents"][0][i][:300].replace("\n", " ")[:300])
    print()
