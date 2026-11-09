import os
import requests
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise RuntimeError("缺少环境变量 DASHSCOPE_API_KEY，请在项目根目录 .env 中配置")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

candidates = [
    "qwen-plus",
    "qwen-max",
    "qwen-turbo",
    "qwen3-plus",
    "qwen3-max",
    "qwen-plus-latest",
    "qwen-max-latest",
]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

for m in candidates:
    try:
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={"model": m, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
            headers=headers,
            timeout=30,
        )
        if r.status_code == 200:
            print(f"{m:20s} -> 200 OK")
        else:
            print(f"{m:20s} -> {r.status_code} {r.text[:120]}")
    except Exception as e:
        print(f"{m:20s} -> ERROR {e}")
