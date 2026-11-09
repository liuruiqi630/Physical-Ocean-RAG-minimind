import json
import time
from pathlib import Path

import numpy as np
import os
import requests
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise RuntimeError("缺少环境变量 DASHSCOPE_API_KEY，请在项目根目录 .env 中配置")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "text-embedding-v4"     # 1024 维；可选 text-embedding-v1/v2(1536)、v3(1024)
BATCH_SIZE = 10                 # 每批文本数

IN_JSON = Path(r"d:\minimind\Physical Oceanography RAG\markdown_chunks_enriched.json")
OUT_NPY = Path(r"d:\minimind\Physical Oceanography RAG\embeddings.npy")


def embed_batch(texts, retries=3):
    """调用 embedding 接口，返回向量列表（顺序与输入一致）"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": MODEL, "input": texts}

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{BASE_URL}/embeddings", json=payload, headers=headers, timeout=60
            )
            if r.status_code == 200:
                data = r.json()
                # 按 index 排序，确保顺序与输入一致
                items = sorted(data["data"], key=lambda d: d["index"])
                return [d["embedding"] for d in items]
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"embedding 失败: {last_err}")


def main():
    chunks = json.loads(IN_JSON.read_text(encoding="utf-8"))
    chunks.sort(key=lambda c: c["chunk_id"])
    texts = [c["contextual_text"] for c in chunks]
    n = len(texts)
    print(f"共 {n} 个 chunk，模型 {MODEL}，批量 {BATCH_SIZE}")

    embeddings = []
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        vecs = embed_batch(batch)
        embeddings.extend(vecs)
        done = min(i + BATCH_SIZE, n)
        print(f"进度: {done}/{n}")

    arr = np.array(embeddings, dtype=np.float32)
    np.save(OUT_NPY, arr)
    print(f"完成：shape={arr.shape}，dtype={arr.dtype}")
    print(f"保存至：{OUT_NPY}")
    print("注：第 i 行对应 chunk_id=i（顺序与 markdown_chunks_enriched.json 一致）")


if __name__ == "__main__":
    main()
