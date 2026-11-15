import argparse
import os
import subprocess
from pathlib import Path
from markdown_it import MarkdownIt


def export_markdown_to_pdf(md_name: str = "DE_CUONG_NLP28.md") -> Path:
    """Xuất một tệp Markdown cạnh script này ra PDF A4, dùng Edge/Chrome headless.

    Chạy không tham số sẽ xuất DE_CUONG_NLP28.md như trước. Truyền tên tệp .md để
    xuất tài liệu khác (ví dụ BAO_CAO nội bộ) mà vẫn giữ nguyên định dạng in.
    """
    base_dir = Path(__file__).parent.resolve()
    md_file = base_dir / md_name
    html_file = md_file.with_suffix(".html")
    pdf_file = md_file.with_suffix(".pdf")

    if not md_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {md_file}")

    md_content = md_file.read_text(encoding="utf-8")
    # Thay thế các ký tự LaTeX đơn giản
    md_content = md_content.replace(r"$\rightarrow$", "→")
    md_content = md_content.replace(r"\rightarrow", "→")

    # Render Markdown sang HTML
    md = MarkdownIt("commonmark").enable("table").enable("strikethrough")
    body_html = md.render(md_content)

    # CSS tinh chỉnh chuẩn in ấn A4 học thuật - Vừa vặn 3 trang đẹp mắt
    css_styles = """
    @page {
        size: A4;
        margin: 14mm 14mm 14mm 14mm;
    }
    * {
        box-sizing: border-box;
    }
    body {
        font-family: "Segoe UI", Arial, Helvetica, sans-serif;
        color: #1e293b;
        line-height: 1.48;
        font-size: 9.8pt;
        background-color: #ffffff;
        margin: 0;
        padding: 0;
    }
    h1 {
        color: #1e3a8a;
        text-align: center;
        font-size: 16pt;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #2563eb;
        padding-bottom: 6px;
    }
    h2 {
        color: #1e40af;
        font-size: 11.8pt;
        font-weight: 700;
        margin-top: 14px;
        margin-bottom: 6px;
        padding: 3px 8px;
        border-left: 4px solid #2563eb;
        background-color: #f1f5f9;
        border-radius: 0 4px 4px 0;
        page-break-after: avoid;
    }
    h3 {
        color: #0f172a;
        font-size: 10.5pt;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 4px;
        page-break-after: avoid;
    }
    p, li {
        text-align: justify;
        margin-bottom: 4px;
    }
    ul, ol {
        padding-left: 20px;
        margin-top: 2px;
        margin-bottom: 6px;
    }
    li {
        margin-bottom: 3px;
    }
    code {
        font-family: Consolas, "Fira Code", Monaco, monospace;
        font-size: 8.8pt;
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 1px 4px;
        border-radius: 3px;
        border: 1px solid #e2e8f0;
    }
    strong {
        color: #0f172a;
    }
    hr {
        border: none;
        border-top: 1px dashed #cbd5e1;
        margin: 10px 0;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 8.0pt;
    }
    th, td {
        border: 1px solid #cbd5e1;
        padding: 5px 6px;
        vertical-align: top;
    }
    th {
        background-color: #1e3a8a;
        color: #ffffff;
        font-weight: 600;
        vertical-align: middle;
    }
    tr:nth-child(even) {
        background-color: #f8fafc;
    }
    tr {
        page-break-inside: avoid;
    }
    """

    full_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Đề cương đề tài NLP-28</title>
<style>
{css_styles}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

    html_file.write_text(full_html, encoding="utf-8")

    # Tìm trình duyệt (Edge hoặc Chrome) để in ra PDF chuẩn xác
    browser_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]

    selected_browser = None
    for b in browser_candidates:
        if os.path.exists(b):
            selected_browser = b
            break

    if not selected_browser:
        raise RuntimeError("Không tìm thấy Microsoft Edge hoặc Google Chrome để in PDF.")

    html_uri = html_file.resolve().as_uri()
    cmd = [
        selected_browser,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_file.resolve()}",
        html_uri,
    ]

    subprocess.run(cmd, capture_output=True, text=True)

    if pdf_file.exists() and pdf_file.stat().st_size > 0:
        print(f"🎉 XUẤT THÀNH CÔNG: {pdf_file.name} ({pdf_file.stat().st_size:,} bytes)")
    else:
        raise RuntimeError("Xuất file PDF thất bại.")

    # Dọn dẹp file html trung gian
    if html_file.exists():
        html_file.unlink()

    return pdf_file


def main():
    parser = argparse.ArgumentParser(description="Xuất Markdown sang PDF chuẩn in A4")
    parser.add_argument(
        "md_files",
        nargs="*",
        help="Tên các tệp .md cần xuất, đặt cạnh script này. Bỏ trống = DE_CUONG_NLP28.md",
    )
    args = parser.parse_args()

    targets = args.md_files or ["DE_CUONG_NLP28.md"]
    for name in targets:
        export_markdown_to_pdf(name)


if __name__ == "__main__":
    main()
