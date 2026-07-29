"""report_html.py 单元测试"""

import os
import tempfile
import pytest


# ── 测试用的 Mini MD ──
TEST_MD = """\
# 📈 每日量化参考 | 测试日报

## 🛡️ ARC 风控: ✅ 正常模式

## 🔥 TID 低位发酵检测

| 子方向 | 申万行业 | 发酵分 | 位置 | 量比 | 20日收益 |
|--------|----------|:------:|:----:|:----:|:--------:|
| 储能 | 电力设备 | 0.85 | 0.92 | 1.8 | +12.3% |
| 小金属 | 有色金属 | 0.72 | 0.65 | 1.7 | +19.9% |

> ⚠️ 低位发酵提醒：小金属、储能等方向出现低位异动

### 选股建议

低位发酵方向中动量前3：ABC科技、DEF材料、GHI能源
"""


@pytest.fixture
def mini_md_file():
    """创建临时 MD 文件，测试结束后自动清理"""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.md', delete=False, encoding='utf-8'
    ) as f:
        f.write(TEST_MD)
        md_path = f.name
    yield md_path
    os.unlink(md_path)


class TestMdToHtml:
    """测试 md_to_html()"""

    def test_creates_file(self, mini_md_file, tmp_path):
        """从测试 MD 生成 HTML，输出文件应存在且 > 0 bytes"""
        from report_html import md_to_html

        out_path = str(tmp_path / 'test_output.html')
        result = md_to_html(mini_md_file, out_path)

        assert result == out_path, "应返回输出路径"
        assert os.path.exists(out_path), "输出文件应存在"
        assert os.path.getsize(out_path) > 0, "输出文件应 > 0 bytes"

    def test_html_contains_keywords(self, mini_md_file, tmp_path):
        """生成的 HTML 应包含 "TID" 或 "低位发酵" 等关键词"""
        from report_html import md_to_html

        out_path = str(tmp_path / 'test_keywords.html')
        md_to_html(mini_md_file, out_path)

        with open(out_path, 'r', encoding='utf-8') as f:
            html = f.read()

        # 关键词检查
        assert 'TID' in html or '低位发酵' in html, \
            f"HTML 应包含 TID 或 低位发酵"

        # 结构检查
        assert '<!DOCTYPE html>' in html
        assert '<article>' in html
        assert '</html>' in html

        # 内容透传检查
        assert '储能' in html
        assert '小金属' in html

    def test_produces_dark_theme_elements(self, mini_md_file, tmp_path):
        """输出 HTML 应包含暗色主题 CSS 变量"""
        from report_html import md_to_html

        out_path = str(tmp_path / 'test_dark.html')
        md_to_html(mini_md_file, out_path)

        with open(out_path, 'r', encoding='utf-8') as f:
            html = f.read()

        # 暗色主题标志
        assert '--bg: #0f1117' in html, "应包含暗色背景变量"

        # 表格渲染检查
        assert '<table>' in html, "应包含表格"
        assert '<thead>' in html, "应包含表头"

        # 警告块
        assert 'warning-block' in html, "应包含警告块 CSS 类"


class TestDefaultPaths:
    """测试默认路径"""

    def test_default_paths(self):
        """md_to_html() 不传参数时使用默认路径（如果 daily_advice.md 存在）"""
        from report_html import md_to_html, DEFAULT_MD_PATH

        if not os.path.exists(DEFAULT_MD_PATH):
            pytest.skip(f"默认 MD 文件不存在: {DEFAULT_MD_PATH}")

        # 输出到临时目录避免覆盖现有 HTML
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, 'daily_advice.html')
            result = md_to_html(DEFAULT_MD_PATH, out_path)

            assert result == out_path
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0


class TestEdgeCases:
    """边界条件测试"""

    def test_simple_markdown(self, tmp_path):
        """测试简单 Markdown（无表格、无引用）"""
        from report_html import md_to_html

        md_path = tmp_path / 'simple.md'
        md_path.write_text("# Hello\n\nThis is **bold** text.\n", encoding='utf-8')

        out_path = str(tmp_path / 'simple.html')
        md_to_html(str(md_path), out_path)

        with open(out_path, 'r', encoding='utf-8') as f:
            html = f.read()

        assert '<h1>Hello</h1>' in html
        assert '<strong>bold</strong>' in html
        assert '--bg: #0f1117' in html  # dark theme applied

    def test_table_only(self, tmp_path):
        """测试纯表格 Markdown"""
        from report_html import md_to_html

        table_md = """\
# 纯表格测试

| 名称 | 分数 | 涨跌 |
|------|:----:|-----:|
| A    | 1.5  | +3.2% |
| B    | 0.8  | -1.1% |
"""
        md_path = tmp_path / 'table.md'
        md_path.write_text(table_md, encoding='utf-8')

        out_path = str(tmp_path / 'table.html')
        md_to_html(str(md_path), out_path)

        with open(out_path, 'r', encoding='utf-8') as f:
            html = f.read()

        assert '<table>' in html
        assert '<thead>' in html
        assert '<tbody>' in html
        assert 'A' in html and 'B' in html

        # 数字列应有对齐样式
        assert 'style="text-align: center;"' in html
        assert 'style="text-align: right;"' in html
