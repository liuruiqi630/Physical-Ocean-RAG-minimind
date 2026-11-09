# -*- coding: utf-8 -*-
"""
RAG 检索增强问答 —— 《物理海洋学笔记》
四步流程：
    第一步：加载已有的 Chroma 数据库
    第二步：执行检索
    第三步：组装 Prompt
    第四步：调用 qwen API 生成回答
"""
import os
import sys
from pathlib import Path

# 离线加载本地 embedding 模型，避免联网与 SSL 问题
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import requests
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv

# ---------------- 路径与常量 ----------------
CHROMA_DIR = Path(r"d:\minimind\Physical Oceanography RAG\chroma_db")
COLLECTION = "wulihaiyang_notes"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
# BGE 检索：查询侧需要加指令前缀，文档侧不加（与建库时一致）
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
TOP_K = 4

# ---------------- qwen API（阿里云百炼 DashScope，OpenAI 兼容） ----------------
load_dotenv(find_dotenv())
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not QWEN_API_KEY:
    raise RuntimeError("缺少环境变量 DASHSCOPE_API_KEY，请在项目根目录 .env 中配置")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-plus"


class RAGPipeline:
    def __init__(self):
        print("=" * 60)
        print("[第一步] 加载 Chroma 数据库 ...")
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.col = self.client.get_collection(COLLECTION)
        print(f"        已加载 collection '{COLLECTION}'，共 {self.col.count()} 条 chunk")

        print("[准备] 加载检索向量模型 (BGE) ...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        print("        检索向量模型就绪")
        print("=" * 60)

    # ---------------- 第二步：检索 ----------------
    def retrieve(self, question: str, top_k: int = TOP_K):
        q_vec = self.embed_model.encode(
            [QUERY_PREFIX + question], normalize_embeddings=True
        ).tolist()
        res = self.col.query(
            query_embeddings=q_vec,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return list(zip(docs, metas, dists))

    # ---------------- 第三步：组装 Prompt ----------------
    @staticmethod
    def build_messages(question: str, retrieved):
        refs = []
        for i, (doc, meta, dist) in enumerate(retrieved, 1):
            path = meta.get("heading_path", "")
            refs.append(f"[{i}] {path}\n{doc.strip()}")
        context = "\n\n".join(refs)

        system = (
            "你是《物理海洋学》课程的助教。请严格依据下面给出的参考资料回答问题，"
            "不要编造；资料中没有的内容请说明“资料中未找到”。"
        )
        user = f"参考资料：\n\n{context}\n\n问题：{question}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ---------------- 第四步：调用 qwen API 生成 ----------------
    def generate(self, question: str, retrieved):
        messages = self.build_messages(question, retrieved)
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": QWEN_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        r = requests.post(
            f"{QWEN_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=120,
        )
        if r.status_code != 200:
            raise RuntimeError(f"qwen API 调用失败 HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    # ---------------- 完整流程 ----------------
    def answer(self, question: str):
        print(f"\n问题: {question}\n")
        print(f"[第二步] 检索 Top {TOP_K} ...")
        retrieved = self.retrieve(question)
        for i, (_, meta, dist) in enumerate(retrieved, 1):
            print(f"  {i}. {meta.get('heading_path', '')}   (距离 {dist:.4f})")

        print("\n[第三步] 组装 Prompt ...")
        print("[第四步] 调用 qwen API 生成回答中 ...\n")
        answer = self.generate(question, retrieved)
        print("回答：")
        print(answer)
        print("-" * 60)
        return answer


def main():
    rag = RAGPipeline()

    # 支持命令行直接传入问题（便于测试）：python rag_pipeline.py "什么是地转流"
    if len(sys.argv) > 1:
        rag.answer(" ".join(sys.argv[1:]))
        return

    print("\n进入交互模式。输入问题回车提问；输入 exit / quit 或直接回车退出。\n")
    while True:
        try:
            q = input("问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        rag.answer(q)


if __name__ == "__main__":
    main()
