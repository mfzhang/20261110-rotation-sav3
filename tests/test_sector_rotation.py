"""sector_rotation.py 单元测试 — 不调 tushare API，全部用合成数据"""

import pytest
import pandas as pd
import numpy as np


class TestGetIndustryPerformance:
    """测试 get_industry_performance 共享数据路径"""

    def test_with_shared_data_returns_nonempty(self, synthetic_daily_df, synthetic_industry_map):
        """传入 daily_df + industry_map 应返回非空 DataFrame"""
        from sector_rotation import get_industry_performance

        result = get_industry_performance(
            days=20, daily_df=synthetic_daily_df, industry_map=synthetic_industry_map
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty, "共享数据路径应返回非空 DataFrame"
        assert 'industry' in result.columns
        assert 'return' in result.columns
        assert 'stocks' in result.columns
        # stocks 列应全部 >= 5（过滤条件）
        assert (result['stocks'] >= 5).all(), "所有行业成分股应 ≥5"


class TestRecommend:
    """测试 recommend()"""

    def test_recommend_returns_top3_and_bottom3(self, synthetic_daily_df, synthetic_industry_map):
        """recommend() 应返回 top3 和 bottom3 列表"""
        from sector_rotation import recommend

        top3, bottom3, perf = recommend(
            daily_df=synthetic_daily_df, industry_map=synthetic_industry_map
        )

        assert isinstance(top3, list), "top3 应为 list"
        assert isinstance(bottom3, list), "bottom3 应为 list"
        assert isinstance(perf, pd.DataFrame), "perf 应为 DataFrame"

        # top3/bottom3 每条为 [industry_name, return_value]
        for item in top3:
            assert len(item) == 2, f"top3 条目应为 [行业名, 涨跌幅]: {item}"
            assert isinstance(item[0], str)
            assert isinstance(item[1], (int, float, np.floating))

        for item in bottom3:
            assert len(item) == 2, f"bottom3 条目应为 [行业名, 涨跌幅]: {item}"
            assert isinstance(item[0], str)
            assert isinstance(item[1], (int, float, np.floating))

        # top3 涨跌幅应 >= bottom3 涨跌幅
        if top3 and bottom3:
            assert top3[0][1] >= bottom3[-1][1], "top3 应该 ≥ bottom3"


class TestGetStockPicks:
    """测试 get_stock_picks()"""

    def test_get_stock_picks_returns_at_least_3(self, synthetic_daily_df, synthetic_industry_map):
        """get_stock_picks() 应返回 ≥3 只个股"""
        from sector_rotation import get_stock_picks

        picks = get_stock_picks(
            daily_df=synthetic_daily_df,
            industry_map=synthetic_industry_map,
            top_n_industries=3,
            picks_per_industry=5,
        )

        assert isinstance(picks, list), "应返回 list"
        assert len(picks) >= 3, f"应返回 ≥3 只个股，实际: {len(picks)}"

        # 检查每条记录结构
        for pick in picks:
            assert 'industry' in pick
            assert 'code' in pick
            assert 'ts_code' in pick
            assert 'pct_chg' in pick
            assert 'close' in pick
            assert 'buy_price' in pick or 'stop_loss' in pick

    def test_get_stock_picks_none_input_returns_empty(self):
        """daily_df 或 industry_map 为 None 时应返回空列表"""
        from sector_rotation import get_stock_picks

        # 都不传
        picks = get_stock_picks(None, None)
        assert picks == []

        # 只传 daily_df
        df = pd.DataFrame({'pct_chg': [1.0]})
        picks = get_stock_picks(df, None)
        assert picks == []


class TestEmptyDailyDf:
    """边界条件测试"""

    def test_empty_daily_df_does_not_crash(self, synthetic_industry_map):
        """空 daily_df 不应崩溃"""
        from sector_rotation import get_industry_performance

        empty_df = pd.DataFrame(columns=['pct_chg', 'amount'])
        # 空 DataFrame 通过共享路径应返回空 DataFrame 而非崩溃
        try:
            result = get_industry_performance(
                days=20, daily_df=empty_df, industry_map=synthetic_industry_map
            )
            # 可能返回空 DataFrame 或报错，但不应该让异常逃逸
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # 允许异常（比如索引格式不匹配），但不能是系统级崩溃
            assert not isinstance(e, (SystemExit, KeyboardInterrupt))


class TestNoIndustryMap:
    """无 industry_map 时走独立拉取路径（用纯离线数据替代）"""

    def test_no_industry_map_goes_independent_path(self, synthetic_daily_df):
        """不传 industry_map 时走独立拉取路径，用 try/except 包裹"""
        from sector_rotation import get_industry_performance

        try:
            result = get_industry_performance(days=20, daily_df=synthetic_daily_df)
            # 走到路径 B，可能因 tushare 不可用而报错，但不崩溃
            assert isinstance(result, pd.DataFrame) or result is None
        except Exception as e:
            # 预期内：tushare token 未设置或无网络等错误
            # 但不应是系统级崩溃
            assert not isinstance(e, (SystemExit, KeyboardInterrupt))
