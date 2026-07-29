#!/usr/bin/env python3
"""
report_html.py — Markdown 日报 → 暗色主题 HTML 转换器
=====================================================
独立模块，读取 /tmp/sav3/output/daily_advice.md，输出
/tmp/sav3/output/daily_advice.html（暗色主题，飞书可直接渲染）。

用法:
    python3 report_html.py                          # 使用默认路径
    python3 report_html.py /path/to/daily_advice.md  # 指定输入
    python3 report_html.py input.md output.html      # 指定输入输出
"""

import re
import sys
import os
from datetime import datetime

import markdown

# ── 默认路径 ──────────────────────────────────────────────
DEFAULT_MD_PATH = "/tmp/sav3/output/daily_advice.md"
DEFAULT_OUT_PATH = "/tmp/sav3/output/daily_advice.html"

# ── CSS 模板 ─────────────────────────────────────────────
CSS = r"""
:root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface-hover: #22253a;
    --border: #2a2d3a;
    --text: #e4e6f0;
    --text-muted: #8b8fa7;
    --accent: #6c8cff;
    --accent-soft: #6c8cff22;
    --green: #4ade80;
    --green-bg: #4ade8018;
    --amber: #fbbf24;
    --amber-bg: #fbbf2418;
    --red: #f87171;
    --red-bg: #f8717118;
    --radius: 10px;
    --radius-sm: 6px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.8;
    -webkit-font-smoothing: antialiased;
}

article {
    max-width: 860px;
    margin: 0 auto;
    padding: 48px 24px 80px;
}

/* ── Headings ── */
h1 {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
    background: linear-gradient(135deg, var(--accent), #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2 {
    font-size: 1.25rem;
    font-weight: 700;
    margin: 44px 0 14px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--border);
    letter-spacing: -0.01em;
}

h3 {
    font-size: 1.05rem;
    font-weight: 700;
    margin: 28px 0 10px;
    color: var(--accent);
}

/* ── Paragraphs & Text ── */
p {
    margin-bottom: 12px;
    color: #c8ccdb;
}

strong {
    color: var(--text);
    font-weight: 700;
}

/* ── Blockquote ── */
blockquote {
    background: var(--surface);
    border-left: 3px solid var(--accent);
    padding: 14px 18px;
    margin: 18px 0;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    font-size: 0.93rem;
    color: #b8bcd4;
}

blockquote strong { color: var(--accent); }

/* ── Tables ── */
.table-wrap {
    overflow-x: auto;
    margin: 22px 0;
    -webkit-overflow-scrolling: touch;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    min-width: 600px;
}

thead th {
    background: var(--surface);
    padding: 10px 14px;
    text-align: left;
    font-weight: 700;
    border-bottom: 2px solid var(--border);
    color: var(--accent);
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    white-space: nowrap;
}

tbody td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
}

tbody tr:nth-child(even) { background: #13161f; }
tbody tr:hover { background: var(--surface-hover); }

/* Right-align numeric columns */
tbody td.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
}

/* ── Details / Summary (fold blocks) ── */
details {
    margin: 16px 0;
}

summary {
    cursor: pointer;
    font-weight: 700;
    color: var(--accent);
    padding: 8px 0;
    font-size: 0.93rem;
    user-select: none;
}

summary:hover { color: #8fa8ff; }

details > *:not(summary) {
    margin-left: 0;
}

details .table-wrap { margin-top: 8px; }

/* ── ARC Status Badge ── */
.arc-status {
    display: inline-block;
    font-weight: 700;
    font-size: 0.88rem;
    padding: 3px 12px;
    border-radius: 99px;
    margin-left: 6px;
}

.arc-status.normal {
    background: var(--green-bg);
    color: var(--green);
}

.arc-status.crisis {
    background: var(--red-bg);
    color: var(--red);
}

/* ── Warning / Note blocks ── */
.warning-block {
    background: var(--amber-bg);
    color: var(--amber);
    padding: 12px 18px;
    border-radius: var(--radius-sm);
    margin: 20px 0;
    font-size: 0.9rem;
    line-height: 1.7;
}

.warning-block strong { color: var(--amber); }

.note-block {
    background: var(--accent-soft);
    color: #a0aef0;
    padding: 12px 18px;
    border-radius: var(--radius-sm);
    margin: 20px 0;
    font-size: 0.9rem;
    line-height: 1.7;
}

.note-block strong { color: var(--accent); }

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 36px 0;
}

/* ── Links ── */
a {
    color: var(--accent);
    text-decoration: none;
}

a:hover { text-decoration: underline; }

/* ── Inline code ── */
code {
    background: var(--surface);
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.88em;
    color: var(--amber);
}

/* ── Footer ── */
.footer-note {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.8rem;
    margin-top: 48px;
}

/* ── Responsive ── */
@media (max-width: 700px) {
    article { padding: 32px 16px 60px; }
    h1 { font-size: 1.4rem; }
    h2 { font-size: 1.1rem; }
    table { font-size: 0.78rem; min-width: 480px; }
    thead th, tbody td { padding: 8px 10px; }
}
"""

# ── Markdown → HTML 转换 ────────────────────────────────


def _add_arc_status(html: str) -> str:
    """检测 ARC 风控行并添加状态徽章"""
    def _replace(match):
        line = match.group(1)
        if "正常" in line:
            line = re.sub(
                r"(正常模式)",
                r'\1 <span class="arc-status normal">✅ 正常</span>',
                line,
            )
        elif "危机" in line or "🚨" in line:
            line = re.sub(
                r"(危机模式)",
                r'\1 <span class="arc-status crisis">🚨 危机</span>',
                line,
            )
        return f"<h2>{line}</h2>"

    return re.sub(r"<h2>(.*?)</h2>", _replace, html, flags=re.DOTALL)


def _wrap_tables(html: str) -> str:
    """为 <table> 添加 table-wrap 容器"""
    return re.sub(
        r"(<table[\s\S]*?</table>)",
        r'<div class="table-wrap">\n\1\n</div>',
        html,
    )


def _promote_warnings(html: str) -> str:
    """将 ⚠️ 警告段落提升为 warning-block"""
    return re.sub(
        r"<p>(⚠️[\s\S]*?)</p>",
        r'<div class="warning-block">\1</div>',
        html,
    )


def md_to_html(md_path: str | None = None, output_path: str | None = None) -> str:
    """
    将 Markdown 日报转换为暗色主题 HTML。

    Args:
        md_path: Markdown 文件路径，默认 /tmp/sav3/output/daily_advice.md
        output_path: HTML 输出路径，默认 /tmp/sav3/output/daily_advice.html

    Returns:
        output_path: 输出的 HTML 文件路径
    """
    if md_path is None:
        md_path = DEFAULT_MD_PATH
    if output_path is None:
        output_path = DEFAULT_OUT_PATH

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # 使用 Python-Markdown 解析，启用表格、 fenced code、details 等扩展
    body_content = markdown.markdown(
        md_content,
        extensions=["extra", "tables", "fenced_code"],
    )

    # 后处理：保持原有样式能力
    body_content = _add_arc_status(body_content)
    body_content = _wrap_tables(body_content)
    body_content = _promote_warnings(body_content)

    # 提取标题（第一个 h1 的文本内容）
    title_match = re.search(r"<h1>(.+?)</h1>", body_content)
    page_title = title_match.group(1) if title_match else "每日量化参考"

    # 清理 HTML 标签用于 <title>
    clean_title = re.sub(r"<[^>]+>", "", page_title)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{clean_title}</title>
<style>{CSS}</style>
</head>
<body>
<article>

{body_content}

<div class="footer-note">
    SA-System · 量化引擎日报 · {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

</article>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# ── CLI 入口 ────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    md_path = args[0] if len(args) >= 1 else DEFAULT_MD_PATH
    out_path = args[1] if len(args) >= 2 else DEFAULT_OUT_PATH

    if not os.path.exists(md_path):
        print(f"❌ 输入文件不存在: {md_path}", file=sys.stderr)
        sys.exit(1)

    result = md_to_html(md_path, out_path)
    print(f"✅ HTML 日报已生成: {result}")
