"""
Unit tests for fermentation_cross.py
Tests: compute_arc_industry_position, cross_check, run_cross_validation, edge cases
"""
import pytest
import pandas as pd
import numpy as np

from fermentation_cross import (
    compute_arc_industry_position,
    cross_check,
    run_cross_validation,
)
import sub_sectors


class TestComputeArcPosition:
    """test_compute_arc_position: compute_arc_industry_position() returns DataFrame with arc_position"""

    def test_compute_arc_position(self, synthetic_daily_df, synthetic_industry_map):
        arc_pos = compute_arc_industry_position(synthetic_daily_df, synthetic_industry_map, lookback=252)

        assert isinstance(arc_pos, pd.DataFrame)
        assert not arc_pos.empty, "Expected non-empty ARC industry position results"

        # 必须有 arc_position 列
        assert 'arc_position' in arc_pos.columns, "Missing arc_position column"
        assert 'n_stocks' in arc_pos.columns, "Missing n_stocks column"

        # arc_position 应该在 [0, 1] 范围
        assert arc_pos['arc_position'].between(0, 1).all(), "arc_position out of [0,1] range"

        # 至少有一些行业
        assert len(arc_pos) >= 1, f"Expected >=1 industries, got {len(arc_pos)}"


class TestCrossCheck:
    """cross_check tests: adds columns, preserves tier"""

    def test_cross_check_adds_columns(self, synthetic_daily_df, synthetic_industry_map, synthetic_ferm_df):
        arc_pos = compute_arc_industry_position(synthetic_daily_df, synthetic_industry_map, lookback=252)

        result = cross_check(synthetic_ferm_df, arc_pos, sub_sectors_module=sub_sectors)

        # 验证增加了 arc_confirm 和 arc_position_industry 列
        assert 'arc_confirm' in result.columns, "Missing arc_confirm column"
        assert 'arc_position_industry' in result.columns, "Missing arc_position_industry column"

        # arc_confirm 是 bool 类型
        assert result['arc_confirm'].dtype == bool, f"Expected bool, got {result['arc_confirm'].dtype}"

        # 行数不变
        assert len(result) == len(synthetic_ferm_df), \
            f"Row count changed: {len(result)} vs {len(synthetic_ferm_df)}"

    def test_cross_check_preserves_tier(self, synthetic_daily_df, synthetic_industry_map, synthetic_ferm_df):
        arc_pos = compute_arc_industry_position(synthetic_daily_df, synthetic_industry_map, lookback=252)

        result = cross_check(synthetic_ferm_df, arc_pos, sub_sectors_module=sub_sectors)

        # 保留 tier 列且值不变
        assert 'tier' in result.columns, "tier column lost"
        assert list(result['tier']) == list(synthetic_ferm_df['tier']), \
            f"tier values changed: {list(result['tier'])} vs {list(synthetic_ferm_df['tier'])}"

        # 保留 sub_sector 和 shenwan
        assert 'sub_sector' in result.columns
        assert 'shenwan' in result.columns

    def test_cross_check_with_shenwan_column(self, synthetic_daily_df, synthetic_industry_map, synthetic_ferm_df):
        """When ferm_df already has shenwan column, cross_check uses it directly."""
        arc_pos = compute_arc_industry_position(synthetic_daily_df, synthetic_industry_map, lookback=252)

        # synthetic_ferm_df has 'shenwan' column
        result = cross_check(synthetic_ferm_df, arc_pos)

        assert 'arc_confirm' in result.columns
        assert 'arc_position_industry' in result.columns

    def test_cross_check_without_shenwan_no_module(self, synthetic_daily_df, synthetic_industry_map, synthetic_ferm_df):
        """When ferm_df has no shenwan and no sub_sectors_module, arc_confirm all False."""
        arc_pos = compute_arc_industry_position(synthetic_daily_df, synthetic_industry_map, lookback=252)

        # Remove shenwan and don't pass sub_sectors_module
        ferm_no_sw = synthetic_ferm_df.drop(columns=['shenwan'])
        result = cross_check(ferm_no_sw, arc_pos)  # no sub_sectors_module

        assert 'arc_confirm' in result.columns
        assert not result['arc_confirm'].any(), "Expected all arc_confirm=False"
        assert result['arc_position_industry'].isna().all(), "Expected all NaN"


class TestRunCrossValidation:
    """test_run_cross_validation: end-to-end pipeline works"""

    def test_run_cross_validation(self, synthetic_daily_df, synthetic_industry_map, synthetic_ferm_df):
        result = run_cross_validation(
            synthetic_daily_df,
            synthetic_industry_map,
            synthetic_ferm_df,
            sub_sectors_module=sub_sectors,
            lookback=252,
        )

        assert isinstance(result, pd.DataFrame)
        assert 'arc_confirm' in result.columns
        assert 'arc_position_industry' in result.columns
        assert len(result) == len(synthetic_ferm_df)


class TestEmptyFermDf:
    """test_empty_ferm_df: empty ferm_df doesn't crash"""

    def test_empty_ferm_df(self, synthetic_daily_df, synthetic_industry_map):
        arc_pos = compute_arc_industry_position(synthetic_daily_df, synthetic_industry_map, lookback=252)
        empty_ferm = pd.DataFrame()

        result = cross_check(empty_ferm, arc_pos, sub_sectors_module=sub_sectors)

        assert result.empty
        assert 'arc_confirm' in result.columns
        assert 'arc_position_industry' in result.columns
        # 列类型应为 bool 和 float
        assert result['arc_confirm'].dtype == bool
        # float 列可能为空时 dtype 为 float64 或 object; 检查不崩溃即可

    def test_empty_ferm_df_empty_arc(self, synthetic_ferm_df):
        """Empty arc_industry_pos with non-empty ferm_df: all confirm=False, position=NaN."""
        empty_arc = pd.DataFrame()
        result = cross_check(synthetic_ferm_df, empty_arc, sub_sectors_module=sub_sectors)

        assert 'arc_confirm' in result.columns
        assert not result['arc_confirm'].any()
        assert result['arc_position_industry'].isna().all()


class TestMissingClose:
    """test_missing_close: no 'close' column returns empty arc_industry_pos"""

    def test_missing_close(self, synthetic_industry_map):
        # 构造没有 close 列的 daily_df
        np.random.seed(99)
        dates = pd.date_range('2025-01-01', periods=300, freq='B')
        codes = [f'{i:06d}.SH' for i in range(10)]
        rows = []
        for d in dates:
            for c in codes:
                rows.append({
                    'date': d, 'code': c,
                    'pct_chg': np.random.normal(0, 2),
                    'amount': np.random.uniform(1e7, 1e8),
                    # 故意不加 'close'
                })
        daily_no_close = pd.DataFrame(rows).set_index(['date', 'code'])

        result = compute_arc_industry_position(daily_no_close, synthetic_industry_map, lookback=252)

        assert isinstance(result, pd.DataFrame)
        assert result.empty, "Expected empty DataFrame when close column missing"
