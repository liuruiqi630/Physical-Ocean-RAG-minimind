import base64
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import os
import requests
from dotenv import load_dotenv, find_dotenv

# 阿里云百炼 DashScope（OpenAI 兼容接口）
load_dotenv(find_dotenv())
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise RuntimeError("缺少环境变量 DASHSCOPE_API_KEY，请在项目根目录 .env 中配置")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3-vl-plus"

MD_PATH = Path(r"d:\minimind\Physical Oceanography RAG\MinerU_markdown_notes_2113145674139602944.md")
OUT_JSON = Path(r"d:\minimind\Physical Oceanography RAG\image_descriptions.json")
TMP_DIR = Path(r"d:\minimind\Physical Oceanography RAG\_md_images")

SYSTEM_PROMPT = (
    "你是一位资深的物理海洋学教授。你的任务是将论文或笔记中的图表"
    "（如图剖面图、流场图、等值线图、温盐深数据等）转化为精准的文字描述，"
    "以便存入学术知识库供后续检索。"
)

USER_PROMPT = """请仔细分析上传的物理海洋学图表，并严格按照以下 Markdown 格式提取信息。如果图中某些信息模糊不清，请如实标注“无法辨认”，绝对不允许编造数据。

1. 核心主题
图表标题：[提取图中的标题，若无标题请根据内容总结]
所属学科：[如：物理海洋学 / 海洋化学 / 海洋气象学]
物理现象：[如：黑潮路径 / 温跃层分布 / 地转流 / 涡旋]

2. 坐标系与图例
X轴：[物理量名称 及 单位（如：经度 / 度E）]
Y轴：[物理量名称 及 单位（如：深度 / m）]
颜色条（若有）：[颜色代表的物理量及数值范围]
图例说明：[提取图例中不同线型/颜色代表的具体含义]

3. 关键特征与趋势（极重要，供检索使用）
分布特征：[描述数据在空间/时间上的整体分布规律，如：随着深度增加，温度逐渐递减]
极值点：[指出图中的最大值、最小值及其出现的具体坐标位置]
物理机制：[结合图表，用专业术语解释其背后的物理机制，如：该现象是由于科里奥利力与压强梯度力的地转平衡导致的]

4. 数据库摘要（用于 RAG 匹配）
一句话总结：用一句话概括这张图最核心的结论。
知识标签：[提取 3-5 个专业关键词，如：温盐深剖面, 地转流, 副热带环流]"""


def extract_images(text: str):
    """按出现顺序提取所有图片 URL，返回 [(index, line_no, url), ...]"""
    pattern = re.compile(r"!\[image\]\((https?://[^)]+)\)")
    result = []
    for i, line in enumerate(text.splitlines(), start=1):
        m = pattern.search(line)
        if m:
            result.append((len(result) + 1, i, m.group(1)))
    return result


def download(url: str, dest: Path, retries: int = 3):
    """下载图片到本地，带重试"""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and r.content:
                dest.write_bytes(r.content)
                return True
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return False


def call_vision(image_path: Path, retries: int = 3):
    """调用 Qwen-VL 识别图片，返回描述文本"""
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    # 根据扩展名推断 mime
    suffix = image_path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            },
        ],
        "max_tokens": 2000,
    }

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions", json=payload, headers=headers, timeout=120
            )
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(3 * (attempt + 1))
    return f"[识别失败] {last_err}"


def describe_one(item):
    idx, line_no, url = item
    img_path = TMP_DIR / f"img_{idx:02d}.jpg"
    TMP_DIR.mkdir(exist_ok=True)

    # 已下载过则复用
    if not img_path.exists():
        if not download(url, img_path):
            return (idx, line_no, url, "[下载失败]")

    desc = call_vision(img_path)
    return (idx, line_no, url, desc)


def main():
    text = MD_PATH.read_text(encoding="utf-8")
    images = extract_images(text)
    print(f"共发现 {len(images)} 张图片")

    # 断点续传：加载已有结果（只保留成功识别的记录，失败/下载失败的需重试）
    done = {}
    if OUT_JSON.exists():
        try:
            for item in json.loads(OUT_JSON.read_text(encoding="utf-8")):
                d = item.get("description", "")
                if not d.startswith("[识别失败]") and not d.startswith("[下载失败]"):
                    done[item["idx"]] = item
        except Exception:
            pass
    print(f"已成功识别 {len(done)} 张，待处理 {len(images) - len(done)} 张")

    todo = [im for im in images if im[0] not in done]
    if not todo:
        print("全部已完成")
        return

    results = list(done.values())
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(describe_one, im): im for im in todo}
        for future in as_completed(futures):
            idx, line_no, url, desc = future.result()
            results.append(
                {"idx": idx, "line_no": line_no, "url": url, "description": desc}
            )
            # 每完成一张就落盘（断点续传）
            results.sort(key=lambda x: x["idx"])
            OUT_JSON.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[{idx}/{len(images)}] 完成")

    print(f"全部完成，共 {len(results)} 张，结果保存在 {OUT_JSON}")


if __name__ == "__main__":
    main()
