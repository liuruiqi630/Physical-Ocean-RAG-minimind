import sys  # 导入 sys 模块，用于读取命令行参数
from pathlib import Path  # 导入 pathlib 的 Path 类，用于更方便地处理文件路径

import fitz  # 导入 PyMuPDF 库（别名 fitz），用于读取和解析 PDF 文件


def extract_images_from_pdf(pdf_path: Path):  # 定义提取图片的函数，参数为 PDF 文件的 Path 路径对象
    if not pdf_path.exists():  # 判断传入的 PDF 文件是否真实存在
        print(f"文件不存在: {pdf_path}")  # 文件不存在时打印提示信息
        return  # 提前结束函数，不再往下执行

    doc = fitz.open(pdf_path)  # 用 PyMuPDF 打开 PDF 文件，返回文档对象
    output_root = pdf_path.parent / f"{pdf_path.stem}_images"  # 构造输出根目录：PDF 所在目录下，以“文件名_images”命名
    output_root.mkdir(exist_ok=True)  # 创建输出根目录；exist_ok=True 表示目录已存在时不报错

    seen_xrefs = set()  # 创建一个集合，用于记录已经处理过的图片 xref，实现去重
    total = 0  # 初始化已提取图片总数计数器

    for page_index in range(len(doc)):  # 遍历 PDF 的每一页，page_index 从 0 开始
        page = doc[page_index]  # 获取当前页对象
        images = page.get_images(full=True)  # 获取当前页上所有图片的引用信息（full=True 包含更多细节）
        if not images:  # 如果当前页没有任何图片
            continue  # 跳过这一页，进入下一页

        page_dir = output_root / f"page_{page_index + 1}"  # 为当前页创建一个按页码命名的子文件夹（页码从 1 开始）
        page_dir.mkdir(exist_ok=True)  # 创建该页码子文件夹；已存在时不报错

        for img_index, img in enumerate(images, start=1):  # 遍历当前页的图片列表，img_index 从 1 开始编号
            xref = img[0]  # 取图片对象的 xref 编号（图片在 PDF 内部的唯一引用号）
            if xref in seen_xrefs:  # 如果这张图片之前已经被提取过
                continue  # 跳过它，避免重复导出同一张图片
            seen_xrefs.add(xref)  # 把当前 xref 加入已处理集合，标记为已提取

            info = doc.extract_image(xref)  # 根据 xref 提取图片的原始数据，返回包含扩展名和二进制数据的字典
            if not info:  # 如果提取结果为空（可能图片损坏或无法提取）
                continue  # 跳过这张图片

            ext = info["ext"]  # 取出图片的原始扩展名（如 png、jpeg、tiff 等）
            data = info["image"]  # 取出图片的二进制内容（bytes）
            out_path = page_dir / f"img_{img_index:02d}.{ext}"  # 构造输出文件路径：img_序号(补零两位).原始扩展名
            out_path.write_bytes(data)  # 把图片二进制数据写入到目标文件
            total += 1  # 成功提取一张，总数加 1
            print(f"已提取: {out_path.relative_to(pdf_path.parent)}")  # 打印已提取图片的相对路径

    doc.close()  # 关闭 PDF 文档，释放资源
    print(f"完成，共提取 {total} 张图片，保存在: {output_root}")  # 打印提取完成的总结信息


if __name__ == "__main__":  # 判断当前脚本是否作为主程序直接运行（而非被 import 导入）
    if len(sys.argv) > 1:  # 如果运行脚本时传入了命令行参数
        pdf_file = Path(sys.argv[1])  # 把第一个命令行参数作为待处理的 PDF 文件路径
    else:  # 如果没有任何命令行参数
        script_dir = Path(__file__).parent  # 获取脚本文件所在的目录
        pdf_file = script_dir / "notes.pdf"  # 默认使用同目录下的 notes.pdf 作为输入文件

    extract_images_from_pdf(pdf_file)  # 调用提取函数，开始执行图片提取
