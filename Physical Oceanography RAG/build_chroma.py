import json
from pathlib import Path

import numpy as np
import chromadb

IN_JSON = Path(r"d:\minimind\Physical Oceanography RAG\markdown_chunks_enriched.json")
EMB_NPY = Path(r"d:\minimind\Physical Oceanography RAG\embeddings_bge.npy")
DB_DIR = Path(r"d:\minimind\Physical Oceanography RAG\chroma_db")
COLLECTION = "wulihaiyang_notes"


def main():
    chunks = json.loads(IN_JSON.read_text(encoding="utf-8"))
    chunks.sort(key=lambda c: c["chunk_id"])
    embeddings = np.load(EMB_NPY)
    assert len(chunks) == len(embeddings), (
        f"chunk 数 {len(chunks)} 与向量数 {len(embeddings)} 不一致，请先运行 embed_chunks_bge.py"
    )

    print(f"共 {len(chunks)} 个 chunk，向量维度 {embeddings.shape[1]}")

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # 幂等重建：删除同名旧 collection
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    col = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(c["chunk_id"]) for c in chunks]
    documents = [c["contextual_text"] for c in chunks]
    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "heading": c["heading"],
            "heading_path": c["heading_path"],
            "level": c["level"],
        }
        for c in chunks
    ]

    col.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    print(f"完成：collection '{COLLECTION}' 共写入 {col.count()} 条")
    print(f"持久化目录：{DB_DIR}")


if __name__ == "__main__":
    main()
