import json
from pathlib import Path

IN_JSON = Path(r"d:\minimind\Physical Oceanography RAG\markdown_chunks.json")
OUT_JSON = Path(r"d:\minimind\Physical Oceanography RAG\markdown_chunks_enriched.json")


def main():
    chunks = json.loads(IN_JSON.read_text(encoding="utf-8"))

    for c in chunks:
        # 上下文增强：将标题路径前置拼接到内容前，形成长字符串
        heading_path = c["heading_path"]
        content = c["content"]
        c["contextual_text"] = f"{heading_path}\n\n{content}"

    OUT_JSON.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"完成：共 {len(chunks)} 个 chunk 已生成上下文增强文本")
    print(f"结果保存至：{OUT_JSON}")


if __name__ == "__main__":
    main()
