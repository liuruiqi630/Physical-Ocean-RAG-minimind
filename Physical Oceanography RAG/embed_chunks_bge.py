import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"   # 智源 BGE 中文小模型，512 维
IN_JSON = Path(r"d:\minimind\Physical Oceanography RAG\markdown_chunks_enriched.json")
OUT_NPY = Path(r"d:\minimind\Physical Oceanography RAG\embeddings_bge.npy")


def main():
    print(f"加载模型 {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    chunks = json.loads(IN_JSON.read_text(encoding="utf-8"))
    chunks.sort(key=lambda c: c["chunk_id"])
    texts = [c["contextual_text"] for c in chunks]
    print(f"共 {len(texts)} 个 chunk，开始向量化 ...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,   # L2 归一化，便于点积/余弦相似度
        batch_size=32,
        show_progress_bar=True,
    )

    arr = np.asarray(embeddings, dtype=np.float32)
    np.save(OUT_NPY, arr)
    print(f"完成：shape={arr.shape}，dtype={arr.dtype}，维度={arr.shape[1]}")
    print(f"保存至：{OUT_NPY}")


if __name__ == "__main__":
    main()
