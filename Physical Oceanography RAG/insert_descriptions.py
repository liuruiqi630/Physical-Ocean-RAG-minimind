import json
import shutil
from pathlib import Path

MD_PATH = Path(r"d:\minimind\Physical Oceanography RAG\MinerU_markdown_notes_2113145674139602944.md")
OUT_JSON = Path(r"d:\minimind\Physical Oceanography RAG\image_descriptions.json")
BACKUP = Path(r"d:\minimind\Physical Oceanography RAG\MinerU_markdown_notes_2113145674139602944.md.bak")


def build_block(desc: str) -> str:
    """把多行描述转成 Markdown 引用块"""
    lines = desc.split("\n")
    out = ["> **图注**", ">"]
    for line in lines:
        if line.strip():
            out.append("> " + line)
        else:
            out.append(">")
    return "\n".join(out)


def main():
    text = MD_PATH.read_text(encoding="utf-8")
    descs = json.loads(OUT_JSON.read_text(encoding="utf-8"))

    # 备份原文件
    shutil.copy2(MD_PATH, BACKUP)

    inserted = 0
    missing = []
    for item in descs:
        url = item["url"]
        desc = item["description"]
        old = f"![image]({url})"
        if old not in text:
            missing.append(item["idx"])
            continue
        new = old + "\n\n" + build_block(desc)
        text = text.replace(old, new, 1)
        inserted += 1

    MD_PATH.write_text(text, encoding="utf-8")
    print(f"完成：成功插入 {inserted}/{len(descs)} 条")
    if missing:
        print("未找到对应 URL 的 idx:", missing)
    print(f"原文件已备份到: {BACKUP}")


if __name__ == "__main__":
    main()
