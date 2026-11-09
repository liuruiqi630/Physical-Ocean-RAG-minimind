# -*- coding: utf-8 -*-
"""
物理海洋学 RAG 问答 —— Streamlit 版
流程：加载 Chroma -> 检索 -> 组装 Prompt -> 调用 qwen API 生成回答

运行：
    .venv\Scripts\streamlit.exe run streamlit_app.py
"""
import os
from pathlib import Path

# 离线加载本地 embedding 模型，避免联网与 SSL 问题
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import requests
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
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


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """首次加载 Chroma + BGE 向量模型，之后缓存复用。"""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(COLLECTION)
    embed = SentenceTransformer(EMBED_MODEL)
    return col, embed


# ---------------- 第二步：检索 ----------------
def retrieve(col, embed, question: str, top_k: int = TOP_K):
    q_vec = embed.encode(
        [QUERY_PREFIX + question], normalize_embeddings=True
    ).tolist()
    res = col.query(
        query_embeddings=q_vec,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return list(zip(docs, metas, dists))


# ---------------- 第三步：组装 Prompt ----------------
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
def generate(question: str, retrieved):
    messages = build_messages(question, retrieved)
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


# ---------------- UI ----------------
st.set_page_config(page_title="物理海洋学 RAG 问答", layout="centered")
st.title("物理海洋学 RAG 问答")
st.caption("基于课程笔记的检索增强问答 · Chroma 检索 + qwen 生成")

top_k = st.sidebar.slider("检索条数 Top-K", min_value=1, max_value=10, value=TOP_K)
st.sidebar.caption(f"向量模型：{EMBED_MODEL}")
st.sidebar.caption(f"生成模型：{QWEN_MODEL}")

with st.form("qa_form"):
    question = st.text_input(
        "问题",
        placeholder="例如：什么是地转流？它的平衡机制是什么？",
    )
    submitted = st.form_submit_button("提问", type="primary")

if submitted:
    if not question.strip():
        st.warning("请输入问题后再提问")
    else:
        with st.spinner("检索并生成中，请稍候…"):
            col, embed = load_pipeline()
            retrieved = retrieve(col, embed, question.strip(), top_k)
            answer = generate(question.strip(), retrieved)

        st.markdown("### 回答")
        st.markdown(answer)

        with st.expander("参考来源"):
            for i, (doc, meta, dist) in enumerate(retrieved, 1):
                st.markdown(
                    f"**[{i}] {meta.get('heading_path', '')}**　`距离 {dist:.4f}`"
                )
                st.text(doc[:300])
