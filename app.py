# -*- coding: utf-8 -*-
"""
物理海洋学 RAG 问答 —— 网页版
自包含 Flask 应用：加载 Chroma -> 检索 -> 组装 Prompt -> 调用 qwen API 生成回答

运行：
    python app.py
然后浏览器打开 http://127.0.0.1:5000
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
from flask import Flask, request, jsonify, render_template_string
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

app = Flask(__name__)

# 懒加载：首次提问时才初始化（避免启动即占内存/联网）
_pipeline = {"col": None, "embed": None}


def get_pipeline():
    if _pipeline["col"] is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_collection(COLLECTION)
        embed = SentenceTransformer(EMBED_MODEL)
        _pipeline["col"] = col
        _pipeline["embed"] = embed
        print(f"[init] Chroma 已加载：{col.count()} 条 chunk；BGE 已加载")
    return _pipeline["col"], _pipeline["embed"]


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


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>物理海洋学 RAG 问答</title>
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  :root {
    --bg: #f4f7fb;
    --card: #ffffff;
    --primary: #2563eb;
    --text: #1f2937;
    --muted: #6b7280;
    --border: #e5e7eb;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }
  .header {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: #fff;
    padding: 28px 20px;
    text-align: center;
  }
  .header h1 { margin: 0 0 6px; font-size: 24px; }
  .header p { margin: 0; opacity: .9; font-size: 14px; }
  .container { max-width: 860px; margin: 0 auto; padding: 20px; }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
  }
  textarea {
    width: 100%;
    min-height: 96px;
    padding: 12px;
    font-size: 15px;
    font-family: inherit;
    border: 1px solid var(--border);
    border-radius: 8px;
    resize: vertical;
  }
  textarea:focus { outline: none; border-color: var(--primary); }
  .row { display: flex; gap: 10px; margin-top: 12px; align-items: center; }
  button {
    background: var(--primary);
    color: #fff;
    border: none;
    padding: 10px 22px;
    font-size: 15px;
    border-radius: 8px;
    cursor: pointer;
  }
  button:hover { background: #1d4ed8; }
  button:disabled { opacity: .6; cursor: not-allowed; }
  .hint { color: var(--muted); font-size: 13px; }
  .answer { white-space: pre-wrap; font-size: 15px; }
  .answer h1, .answer h2, .answer h3 { margin: 12px 0 6px; }
  .answer code { background: #f1f5f9; padding: 1px 5px; border-radius: 4px; }
  .answer strong { color: #111827; }
  .sources-title { font-weight: 600; margin: 0 0 8px; }
  .source {
    border-left: 3px solid var(--primary);
    padding: 6px 12px;
    margin-bottom: 8px;
    background: #f8fafc;
    border-radius: 0 6px 6px 0;
    font-size: 13px;
  }
  .source .path { font-weight: 600; color: #1d4ed8; }
  .source .dist { color: var(--muted); font-size: 12px; margin-left: 6px; }
  .spinner {
    display: none;
    color: var(--muted);
    font-size: 14px;
  }
</style>
</head>
<body>
  <div class="header">
    <h1>物理海洋学 RAG 问答</h1>
    <p>基于课程笔记的检索增强问答 · 检索 + qwen 生成</p>
  </div>
  <div class="container">
    <div class="card">
      <textarea id="question" placeholder="请输入你的问题，例如：什么是地转流？它的平衡机制是什么？"></textarea>
      <div class="row">
        <button id="askBtn" onclick="ask()">提问</button>
        <span class="spinner" id="spinner">检索并生成中，请稍候…</span>
      </div>
    </div>

    <div class="card" id="answerCard" style="display:none;">
      <div class="sources-title">回答</div>
      <div class="answer" id="answer"></div>
    </div>

    <div class="card" id="sourcesCard" style="display:none;">
      <div class="sources-title">参考来源</div>
      <div id="sources"></div>
    </div>
  </div>

  <script>
    function escapeHtml(s) {
      return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
    function formatAnswer(text) {
      let html = escapeHtml(text);
      html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
      return html;
    }
    async function ask() {
      const q = document.getElementById('question').value.trim();
      if (!q) return;
      const btn = document.getElementById('askBtn');
      const spinner = document.getElementById('spinner');
      btn.disabled = true;
      spinner.style.display = 'inline';
      document.getElementById('answerCard').style.display = 'none';
      document.getElementById('sourcesCard').style.display = 'none';
      try {
        const resp = await fetch('/ask', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({question: q})
        });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);

        const ansEl = document.getElementById('answer');
        ansEl.innerHTML = formatAnswer(data.answer);
        document.getElementById('answerCard').style.display = 'block';

        const srcEl = document.getElementById('sources');
        srcEl.innerHTML = '';
        (data.sources || []).forEach((s, i) => {
          const div = document.createElement('div');
          div.className = 'source';
          div.innerHTML = '<span class="path">[' + (i + 1) + '] ' + escapeHtml(s.heading_path) +
            '</span><span class="dist">距离 ' + s.distance.toFixed(4) + '</span>' +
            '<div>' + escapeHtml(s.snippet || '') + '</div>';
          srcEl.appendChild(div);
        });
        document.getElementById('sourcesCard').style.display = 'block';

        if (window.MathJax && MathJax.typesetPromise) {
          await MathJax.typesetPromise([ansEl]);
        }
      } catch (e) {
        const ansEl = document.getElementById('answer');
        ansEl.innerHTML = '<span style="color:#dc2626;">出错：' + escapeHtml(e.message || String(e)) + '</span>';
        document.getElementById('answerCard').style.display = 'block';
      } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
      }
    }
    document.getElementById('question').addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) ask();
    });
  </script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    try:
        col, embed = get_pipeline()
        retrieved = retrieve(col, embed, question)
        answer = generate(question, retrieved)

        sources = [
            {
                "heading_path": meta.get("heading_path", ""),
                "distance": float(dist),
                "snippet": (doc or "")[:300],
            }
            for doc, meta, dist in retrieved
        ]
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
