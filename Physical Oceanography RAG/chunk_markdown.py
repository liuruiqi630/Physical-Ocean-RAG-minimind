import json
import re
from pathlib import Path

MD_PATH = Path(r"d:\minimind\Physical Oceanography RAG\MinerU_markdown_notes_2113145674139602944.md")
OUT_JSON = Path(r"d:\minimind\Physical Oceanography RAG\markdown_chunks.json")

CHUNK_SIZE = 512        # 目标 chunk 大小（token）
OVERLAP = 51            # 块间重叠（token，约 10%）


# ---------------------------------------------------------------
# token 估算：中文 1 字 = 1 token，英文/数字/公式 4 字符 = 1 token
# ---------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    han = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - han
    return han + other // 4


# ---------------------------------------------------------------
# 标题层级：带编号（1 / 1.1 / 1.1.1）按点数定深度；无编号视为子标题
# ---------------------------------------------------------------
def heading_level(title: str) -> int:
    m = re.match(r"^(\d+(?:\.\d+)*)", title)
    if m:
        return m.group(1).count(".") + 1
    return 0


# ---------------------------------------------------------------
# 按标题切分文档为 sections，构建完整标题路径
# ---------------------------------------------------------------
def split_by_headings(text: str):
    lines = text.splitlines()
    sections = []
    stack = []

    pre_lines = []
    body_started = False
    current_heading = None
    current_lines = []

    for line in lines:
        m = re.match(r"^##\s+(.*)$", line)
        if m:
            body_started = True
            if current_heading is not None:
                sections.append({
                    "heading": current_heading,
                    "path": " > ".join(t for _, t in stack),
                    "level": stack[-1][0] if stack else 0,
                    "content": "\n".join(current_lines),
                })
            title = m.group(1).strip()
            lvl = heading_level(title)
            if lvl == 0:
                parent_lvl = stack[-1][0] if stack else 0
                lvl = parent_lvl + 1
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, title))
            current_heading = title
            current_lines = [line]
        else:
            if body_started:
                current_lines.append(line)
            else:
                pre_lines.append(line)

    if current_heading is not None:
        sections.append({
            "heading": current_heading,
            "path": " > ".join(t for _, t in stack),
            "level": stack[-1][0] if stack else 0,
            "content": "\n".join(current_lines),
        })

    if pre_lines:
        sections.insert(0, {
            "heading": "目录",
            "path": "物理海洋学笔记",
            "level": 0,
            "content": "\n".join(pre_lines),
        })

    return sections


# ---------------------------------------------------------------
# 把内容拆为最小单元（公式块保持整体，普通文本按段落/行拆分）
# ---------------------------------------------------------------
def split_into_units(content: str):
    parts = re.split(r"(\$\$.*?\$\$)", content, flags=re.DOTALL)
    units = []
    for p in parts:
        if p == "":
            continue
        if p.startswith("$$"):
            units.append(p)                       # 公式块整体
        else:
            for para in re.split(r"\n\n+", p):
                if para.strip():
                    units.append(para + "\n\n")
    return units


# ---------------------------------------------------------------
# 把超限普通文本进一步按行、按字符硬切到 <= chunk_size
# ---------------------------------------------------------------
def hard_split(text: str, chunk_size: int):
    result = []
    for line in text.split("\n"):
        if estimate_tokens(line) <= chunk_size:
            result.append(line + "\n")
        else:
            # 单行仍超限：按约 chunk_size*2 字符硬切，尽量在标点处断
            char_lim = chunk_size * 2
            while estimate_tokens(line) > chunk_size:
                cut = line[:char_lim]
                idx = max(cut.rfind("。"), cut.rfind("；"), cut.rfind("，"),
                          cut.rfind(" "), cut.rfind(","))
                if idx < char_lim * 0.3:
                    idx = char_lim
                result.append(line[:idx + 1] + "\n")
                line = line[idx + 1:]
            if line.strip():
                result.append(line + "\n")
    return result


# ---------------------------------------------------------------
# 超长章节按固定大小细分（带重叠）
# ---------------------------------------------------------------
def split_long_content(content: str, chunk_size: int, overlap: int):
    units = split_into_units(content)

    # 预拆分：非公式单元若超限，进一步按行/字符切分
    refined = []
    for u in units:
        if u.startswith("$$"):
            refined.append(u)                       # 公式块保持整体（不拆）
        elif estimate_tokens(u) > chunk_size:
            refined.extend(hard_split(u, chunk_size))
        else:
            refined.append(u)

    # 贪心累积 + 重叠回退
    chunks = []
    current = []
    current_tokens = 0
    for u in refined:
        ut = estimate_tokens(u)
        if current_tokens + ut > chunk_size and current:
            chunks.append("".join(current))
            # 回退 overlap 部分作为下一块开头
            overlap_units = []
            overlap_tokens = 0
            for uu in reversed(current):
                t = estimate_tokens(uu)
                if overlap_tokens + t > overlap:
                    break
                overlap_units.insert(0, uu)
                overlap_tokens += t
            current = overlap_units
            current_tokens = overlap_tokens
        current.append(u)
        current_tokens += ut

    if current:
        chunks.append("".join(current))
    return chunks


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------
def main():
    text = MD_PATH.read_text(encoding="utf-8")
    sections = split_by_headings(text)
    print(f"按标题切分为 {len(sections)} 个章节")

    chunks = []
    chunk_id = 0

    for sec in sections:
        sec_tokens = estimate_tokens(sec["content"])
        if sec_tokens <= CHUNK_SIZE:
            chunks.append({
                "chunk_id": chunk_id,
                "heading": sec["heading"],
                "heading_path": sec["path"],
                "level": sec["level"],
                "token_count": sec_tokens,
                "char_count": len(sec["content"]),
                "content": sec["content"],
            })
            chunk_id += 1
        else:
            sub_chunks = split_long_content(sec["content"], CHUNK_SIZE, OVERLAP)
            for i, sc in enumerate(sub_chunks):
                chunks.append({
                    "chunk_id": chunk_id,
                    "heading": sec["heading"],
                    "heading_path": sec["path"],
                    "level": sec["level"],
                    "sub_index": i,
                    "token_count": estimate_tokens(sc),
                    "char_count": len(sc),
                    "content": sc,
                })
                chunk_id += 1

    OUT_JSON.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_tokens = sum(c["token_count"] for c in chunks)
    print(f"完成：共 {len(chunks)} 个 chunk，总 {total_tokens} token")
    print(f"结果保存至：{OUT_JSON}")


if __name__ == "__main__":
    main()
