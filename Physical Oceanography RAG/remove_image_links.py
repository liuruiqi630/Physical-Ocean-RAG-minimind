import re
from pathlib import Path

MD_PATH = Path(r"d:\minimind\Physical Oceanography RAG\MinerU_markdown_notes_2113145674139602944.md")


def main():
    text = MD_PATH.read_text(encoding="utf-8")

    # 匹配图片引用（![xxx](url) 或空占位 ![]()），统一替换为 ![image]()，保留 image 单词、去掉 URL
    pattern = re.compile(r"!\[[^\]]*\]\([^)]*\)")
    new_text, count = pattern.subn("![image]()", text)

    MD_PATH.write_text(new_text, encoding="utf-8")
    print(f"完成：共替换 {count} 处图片引用为 ![image]()")


if __name__ == "__main__":
    main()
