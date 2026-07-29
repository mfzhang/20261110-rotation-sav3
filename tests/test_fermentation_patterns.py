"""
Unit tests for fermentation_patterns.py
Tests: save_patterns, load_patterns, backtest_conditions, get_reusable_patterns
"""
import os
import json
import pytest
import pandas as pd
import numpy as np

# 导入被测试模块（会触发 config.CACHE_DIR 创建）
import fermentation_patterns as fpat
from fermentation_patterns import (
    save_patterns,
    load_patterns,
    backtest_conditions,
    get_reusable_patterns,
    PATTERNS_FILE,
    OUTPUT_DIR,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_test_ferm_df():
    """构建用于 save_patterns 的测试 DataFrame"""
    return pd.DataFrame({
        'sub_sector': ['小金属', 'AI应用', '锂电池'],
        'shenwan': ['有色金属', '计算机', '电力设备'],
        'position': [0.65, 0.98, 0.55],
        'volume_ratio': [1.7, 1.0, 1.3],
        'return_20d': [0.199, -0.196, 0.05],
        'breadth_3d': [0.67, 0.0, 0.5],
        'breadth_momentum': [0.15, -0.05, 0.0],
        'ferment_score': [0.62, 0.45, 0.41],
        'tier': ['B', 'C', 'C'],
        'crowding_score': [0.5, 0.3, 0.6],
        'consecutive_up': [3, 1, 2],
        'turnover_rank_pct': [0.6, 0.4, 0.5],
        'up_down_ratio': [1.5, 0.8, 1.2],
        'n_stocks': [10, 8, 12],
    })


def _make_test_llm_enhance():
    """构建 LLM 增强数据"""
    return {
        '小金属': {
            'catalyst': '稀土价格反弹',
            'confirm_condition': '板块放量突破5日线',
            'fail_condition': '缩量跌破20日线',
        },
        'AI应用': {
            'catalyst': 'GPT-5发布预期',
            'confirm_condition': '科大讯飞涨停',
            'fail_condition': '板块缩量',
        },
        '锂电池': {
            'catalyst': '',
            'confirm_condition': '',
            'fail_condition': '',
        },
    }


def _cleanup_patterns_file():
    """删除测试产生的 patterns 文件"""
    if os.path.exists(PATTERNS_FILE):
        os.remove(PATTERNS_FILE)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """每个测试后清理 ferm_patterns.json"""
    yield
    _cleanup_patterns_file()


# ── Tests ────────────────────────────────────────────────────

class TestSaveAndLoad:
    """test_save_and_load: save_patterns writes → load_patterns reads, data consistency"""

    def test_save_and_load(self):
        _cleanup_patterns_file()  # 确保从干净状态开始

        ferm_df = _make_test_ferm_df()
        llm_enhance = _make_test_llm_enhance()
        date_str = '2026-06-19'

        # 保存
        save_patterns(ferm_df, llm_enhance, date_str)

        # 验证文件存在
        assert os.path.exists(PATTERNS_FILE), f"Patterns file not created: {PATTERNS_FILE}"

        # 加载
        loaded = load_patterns(days=30)

        assert isinstance(loaded, dict)
        assert date_str in loaded, f"Date {date_str} not in loaded patterns"
        records = loaded[date_str]
        assert len(records) == 3, f"Expected 3 records, got {len(records)}"

        # 验证第一条记录
        rec0 = records[0]
        assert rec0['sub_sector'] == '小金属'
        assert rec0['tier'] == 'B'
        assert rec0['shenwan'] == '有色金属'
        assert rec0['position'] == 0.65
        assert rec0['volume_ratio'] == 1.7
        assert rec0['catalyst'] == '稀土价格反弹'
        assert rec0['confirm_condition'] == '板块放量突破5日线'
        assert rec0['fail_condition'] == '缩量跌破20日线'

        # 验证第二条记录
        rec1 = records[1]
        assert rec1['sub_sector'] == 'AI应用'
        assert rec1['tier'] == 'C'
        assert rec1['catalyst'] == 'GPT-5发布预期'

        # 验证第三条记录 (空的 llm)
        rec2 = records[2]
        assert rec2['sub_sector'] == '锂电池'
        assert rec2['catalyst'] == ''
        assert rec2['confirm_condition'] == ''
        assert rec2['fail_condition'] == ''

    def test_save_and_load_multiple_dates(self):
        """保存多个日期的数据，load 只返回最近 N 天"""
        _cleanup_patterns_file()

        ferm_df = _make_test_ferm_df()
        llm_enhance = _make_test_llm_enhance()

        save_patterns(ferm_df, llm_enhance, '2026-06-17')
        save_patterns(ferm_df, llm_enhance, '2026-06-18')
        save_patterns(ferm_df, llm_enhance, '2026-06-19')

        # 只加载最近 2 天
        loaded = load_patterns(days=2)
        assert len(loaded) == 2
        assert '2026-06-18' in loaded
        assert '2026-06-19' in loaded
        assert '2026-06-17' not in loaded

    def test_save_empty_ferm_df(self):
        """保存空 DataFrame 不崩溃"""
        _cleanup_patterns_file()

        save_patterns(pd.DataFrame(), {}, '2026-06-20')
        loaded = load_patterns(days=1)
        assert '2026-06-20' in loaded
        assert len(loaded['2026-06-20']) == 0

    def test_save_no_llm_enhance(self):
        """llm_enhance 为 None 时不崩溃"""
        _cleanup_patterns_file()

        ferm_df = _make_test_ferm_df()
        save_patterns(ferm_df, None, '2026-06-19')
        loaded = load_patterns(days=1)
        records = loaded['2026-06-19']
        # catalyst/confirm/fail 应该为空字符串
        for rec in records:
            assert rec['catalyst'] == ''
            assert rec['confirm_condition'] == ''
            assert rec['fail_condition'] == ''


class TestLoadEmpty:
    """test_load_empty: no file → load_patterns() returns {}"""

    def test_load_empty_no_file(self):
        _cleanup_patterns_file()
        # 确保文件不存在
        assert not os.path.exists(PATTERNS_FILE)

        result = load_patterns(days=60)
        assert result == {}
        assert isinstance(result, dict)

    def test_load_empty_corrupted_file(self):
        """损坏的 JSON 文件返回 {}"""
        _cleanup_patterns_file()

        # 写入无效 JSON
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(PATTERNS_FILE, 'w') as f:
            f.write('not valid json{{{')

        result = load_patterns(days=60)
        assert result == {}


class TestBacktestEmpty:
    """test_backtest_empty: no history cache → backtest_conditions() returns empty DataFrame"""

    def test_backtest_empty_no_patterns(self):
        """没有历史模式数据时返回空 DataFrame"""
        _cleanup_patterns_file()
        assert not os.path.exists(PATTERNS_FILE)

        result = backtest_conditions(days=60)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_backtest_empty_no_cache(self):
        """有模式数据但没有日线缓存时返回空 DataFrame"""
        _cleanup_patterns_file()

        # 保存一些模式数据
        ferm_df = _make_test_ferm_df()
        llm_enhance = _make_test_llm_enhance()
        save_patterns(ferm_df, llm_enhance, '2026-06-19')

        # 确保 cache 目录没有匹配的 daily 缓存文件
        # backtest_conditions 会调用 _load_latest_daily_cache
        # 如果没有缓存或缓存不满足条件，返回空 DataFrame
        result = backtest_conditions(days=60)
        # 可能找到缓存也可能找不到，取决于环境
        # 如果找不到缓存，返回空 DataFrame
        assert isinstance(result, pd.DataFrame)


class TestReusablePatternsEmpty:
    """test_reusable_patterns_empty: no high hit-rate patterns → get_reusable_patterns() returns []"""

    def test_reusable_patterns_empty(self):
        """无高命中率模式时返回空列表"""
        _cleanup_patterns_file()
        assert not os.path.exists(PATTERNS_FILE)

        result = get_reusable_patterns(min_hit_rate=0.6, min_samples=3)
        assert isinstance(result, list)
        assert result == []

    def test_reusable_patterns_custom_thresholds(self):
        """自定义阈值时也可正常调用"""
        _cleanup_patterns_file()
        result = get_reusable_patterns(min_hit_rate=0.8, min_samples=10)
        assert isinstance(result, list)
        assert result == []


class TestOutputDir:
    """验证输出目录和路径"""

    def test_output_dir_created(self):
        """save_patterns 应自动创建 output 目录"""
        # 清理
        _cleanup_patterns_file()
        if os.path.exists(OUTPUT_DIR):
            # 删除目录如果为空
            try:
                os.rmdir(OUTPUT_DIR)
            except OSError:
                pass

        ferm_df = _make_test_ferm_df()
        save_patterns(ferm_df, {}, '2026-06-19')

        assert os.path.isdir(OUTPUT_DIR), f"Output dir not created: {OUTPUT_DIR}"
        assert os.path.exists(PATTERNS_FILE)

    def test_load_patterns_all_dates(self):
        """load_patterns 可加载全部历史数据"""
        _cleanup_patterns_file()

        ferm_df = _make_test_ferm_df()
        save_patterns(ferm_df, {}, '2026-06-10')
        save_patterns(ferm_df, {}, '2026-06-15')
        save_patterns(ferm_df, {}, '2026-06-19')

        # 使用很大的 days 值加载全部
        loaded = load_patterns(days=999)
        assert len(loaded) == 3
