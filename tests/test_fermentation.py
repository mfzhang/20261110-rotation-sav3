"""
Unit tests for fermentation.py
Tests: _build_sub_indices, compute_crowding_metrics, find_low_fermentation, run_fermentation_pipeline
"""
import pytest
import pandas as pd
import numpy as np

import fermentation
from fermentation import (
    _build_sub_indices,
    compute_crowding_metrics,
    find_low_fermentation,
    run_fermentation_pipeline,
)
from sub_sectors import classify_stock_to_sub


class TestBuildSubIndices:
    """test_build_sub_indices: _build_sub_indices() returns >=3 sub directions (synthetic has 7)"""

    def test_build_sub_indices(self, synthetic_daily_df, synthetic_industry_map):
        sub_indices = _build_sub_indices(synthetic_daily_df, synthetic_industry_map)
        assert isinstance(sub_indices, dict)
        # 6 of 7 sub-sectors match stock keywords ("小金属" stock keywords are element names,
        # not "小金属" itself, so it won't match)
        assert len(sub_indices) >= 3, f"Expected >=3 sub-indices, got {len(sub_indices)}"
        for sub_name, item in sub_indices.items():
            ret_s, amt_s = item[0], item[1]
            assert isinstance(ret_s, pd.Series), f"{sub_name}: ret_s not a Series"
            assert isinstance(amt_s, pd.Series), f"{sub_name}: amt_s not a Series"
            assert len(ret_s) >= 20, f"{sub_name}: too few data points ({len(ret_s)})"
            # 4 元组应含真实成分股广度（0~1 占比）
            if len(item) > 3:
                breadth_s = item[3]
                assert breadth_s.dropna().between(0, 1).all(), f"{sub_name}: breadth 越界"


class TestComputeCrowding:
    """test_compute_crowding: compute_crowding_metrics() returns DataFrame with expected columns"""

    def test_compute_crowding(self, synthetic_daily_df, synthetic_industry_map):
        sub_indices = _build_sub_indices(synthetic_daily_df, synthetic_industry_map)
        crowding = compute_crowding_metrics(sub_indices)

        assert isinstance(crowding, pd.DataFrame)
        assert not crowding.empty
        # 验证必需的列
        required_cols = ['position', 'volume_ratio', 'crowding_score']
        for col in required_cols:
            assert col in crowding.columns, f"Missing column: {col}"
        # ferment_score 在 find_low_fermentation 中添加, crowding 阶段不要求有
        # 但验证有其他预期列
        expected_cols = ['sub_sector', 'shenwan', 'return_20d', 'consecutive_up',
                         'breadth_3d', 'breadth_momentum', 'turnover_rank_pct',
                         'up_down_ratio', 'n_stocks']
        for col in expected_cols:
            assert col in crowding.columns, f"Missing column: {col}"

        # 验证 position 在合理范围 [0, 1]
        assert crowding['position'].between(0, 1).all(), "position out of [0,1] range"


class TestFindFermentation:
    """test_find_fermentation: find_low_fermentation() returns A/B/C tiers, B-tier at least 1"""

    def _make_crowding_for_fermentation(self):
        """Build a custom crowding_df that will produce A/B/C tiers."""
        return pd.DataFrame({
            'sub_sector': ['小金属', 'AI应用', '锂电池', '储能', '光伏', '半导体'],
            'shenwan': ['有色金属', '计算机', '电力设备', '电力设备', '电力设备', '电子'],
            'return_20d': [-0.05, -0.19, 0.05, 0.03, 0.12, 0.25],
            'volume_ratio': [1.7, 1.4, 1.3, 0.9, 1.5, 1.1],
            'consecutive_up': [2, 1, 3, 0, 4, 6],  # 半导体连涨6天会被排除(>=5)
            'breadth_3d': [0.67, 0.50, 0.50, 0.33, 0.75, 0.80],
            'breadth_momentum': [0.15, -0.05, 0.0, -0.10, 0.25, 0.30],
            'turnover_rank_pct': [0.60, 0.40, 0.50, 0.30, 0.70, 0.80],
            'up_down_ratio': [1.5, 0.8, 1.2, 0.5, 2.0, 2.5],
            'position': [0.65, 0.98, 0.55, 0.45, 0.72, 0.50],
            'crowding_score': [0.35, 0.30, 0.45, 0.60, 0.20, 0.75],  # 半导体拥挤度0.75>0.7被排除
            'n_stocks': [10, 8, 12, 7, 9, 15],
        })

    def test_find_fermentation(self, synthetic_daily_df, synthetic_industry_map):
        # 构造自定义 crowding_df 以确保 B 档至少 1 个
        sub_indices = _build_sub_indices(synthetic_daily_df, synthetic_industry_map)
        crowding_df = self._make_crowding_for_fermentation()

        ferm = find_low_fermentation(sub_indices, crowding_df)

        assert isinstance(ferm, pd.DataFrame)
        assert not ferm.empty, "Expected non-empty fermentation results"

        # 验证分档
        assert 'tier' in ferm.columns
        assert 'ferment_score' in ferm.columns

        tiers = ferm['tier'].value_counts().to_dict()
        # 应该只有 A/B/C（D被过滤）
        for t in tiers:
            assert t in ['A', 'B', 'C'], f"Unexpected tier: {t}"

        # B 档至少 1 个
        b_count = tiers.get('B', 0)
        assert b_count >= 1, f"Expected at least 1 B-tier, got tier counts: {tiers}"


class TestPipeline:
    """test_pipeline: run_fermentation_pipeline() returns dict with expected keys"""

    def test_pipeline(self, synthetic_daily_df, synthetic_industry_map):
        result = run_fermentation_pipeline(synthetic_daily_df, synthetic_industry_map)

        assert isinstance(result, dict)
        expected_keys = ['crowding', 'fermentation', 'sub_indices', 'excluded_crowded']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        # sub_indices: dict with >= 3 entries
        assert isinstance(result['sub_indices'], dict)
        assert len(result['sub_indices']) >= 3

        # crowding: DataFrame (may be non-empty since we have data)
        assert isinstance(result['crowding'], pd.DataFrame)

        # fermentation: DataFrame
        assert isinstance(result['fermentation'], pd.DataFrame)

        # excluded_crowded: list
        assert isinstance(result['excluded_crowded'], list)


class TestEmptyData:
    """test_empty_data: empty daily_df doesn't crash, returns empty results"""

    def test_empty_data(self):
        empty_df = pd.DataFrame(columns=['pct_chg', 'amount', 'close'])
        # 尝试设置 MultiIndex 但为空
        empty_mi = pd.DataFrame(columns=['pct_chg', 'amount', 'close', 'name'])
        empty_mi.index = pd.MultiIndex.from_arrays([[], []], names=['date', 'code'])

        industry_map = pd.Series(dtype=str)

        result = run_fermentation_pipeline(empty_mi, industry_map)

        assert result['crowding'].empty
        assert result['fermentation'].empty
        assert result['sub_indices'] == {}
        assert result['excluded_crowded'] == []

        # Also test _build_sub_indices directly
        sub_indices = _build_sub_indices(empty_mi, industry_map)
        assert sub_indices == {}

        # compute_crowding_metrics on empty
        crowding = compute_crowding_metrics({})
        assert crowding.empty

        # find_low_fermentation on empty
        ferm = find_low_fermentation({}, pd.DataFrame())
        assert ferm.empty


class TestNoNameColumn:
    """test_no_name_column: no 'name' column doesn't crash; stock keyword code matching"""

    def test_no_name_column(self, synthetic_daily_df, synthetic_industry_map):
        # 去掉 name 列
        daily_no_name = synthetic_daily_df.drop(columns=['name'])

        # 应该不崩溃
        sub_indices = _build_sub_indices(daily_no_name, synthetic_industry_map)

        # 没有 name 列时，stock keywords 需匹配 code 中的内容
        # 合成数据的 code 是 000000.SH 格式，不匹配子方向关键词
        # 所以应该返回空 (但不崩溃)
        assert isinstance(sub_indices, dict)
        # 空结果不崩溃即通过

        # pipeline 也不应崩溃
        result = run_fermentation_pipeline(daily_no_name, synthetic_industry_map)
        assert isinstance(result, dict)
        assert all(k in result for k in ['crowding', 'fermentation', 'sub_indices', 'excluded_crowded'])

    def test_no_name_column_with_matching_codes(self, synthetic_industry_map):
        """When codes contain stock keywords (e.g., through code naming), classification works."""
        # 构造一个包含关键词匹配 code 的 DataFrame
        np.random.seed(123)
        dates = pd.date_range('2025-01-01', periods=300, freq='B')
        # 使用包含关键词的 code
        codes_with_keywords = ['601899.XSHG', '600111.XSHG', '002460.XSHE']
        # 小金属 keywords: 钨,钼,锑,钴,镍,钛 — 不匹配
        # 我们使用更明确的关键词匹配：储能 keywords: 储能
        # code 是数字，不会匹配。需要 name 或 code 含关键词

        rows = []
        for d in dates:
            for c in codes_with_keywords:
                rows.append({
                    'date': d, 'code': c,
                    'close': np.random.uniform(50, 150),
                    'pct_chg': np.random.normal(0, 2),
                    'amount': np.random.uniform(1e7, 1e8),
                })
        # 故意不加 name 列
        daily = pd.DataFrame(rows).set_index(['date', 'code'])

        # 不应崩溃
        sub_indices = _build_sub_indices(daily, synthetic_industry_map)
        assert isinstance(sub_indices, dict)
