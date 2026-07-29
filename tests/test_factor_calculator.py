"""factor_calculator 回归测试 — 防 P0-1 (mdd NaN 污染 low_risk) 复发。

纯合成数据，不调 tushare API。
"""
import numpy as np
import pandas as pd

from factor_calculator import calculate_factors


def _make_daily(n_days=300, n_stocks=20):
    """合成日K：随机游走价格，每只股票首行 ret 必为 NaN（pct_change 期首）。"""
    np.random.seed(7)
    dates = pd.date_range('2025-01-01', periods=n_days, freq='B')
    rows = []
    for i in range(n_stocks):
        code = f'{i:06d}.SH'
        rets = np.random.normal(0.0005, 0.02, n_days)
        close = 100 * (1 + pd.Series(rets)).cumprod().values
        for d, c, r in zip(dates, close, rets):
            rows.append({'date': d, 'code': code, 'close': c,
                         'high': c * 1.01, 'low': c * 0.99, 'ret': r})
    df = pd.DataFrame(rows).set_index(['date', 'code'])
    # 模拟生产管线：首行 ret = NaN
    df['ret'] = df.groupby('code')['close'].pct_change()
    df['high_52w'] = df.groupby('code')['high'].transform(
        lambda x: x.rolling(252, min_periods=1).max())
    df['low_52w'] = df.groupby('code')['low'].transform(
        lambda x: x.rolling(252, min_periods=1).min())
    idx_ret = pd.Series(np.random.normal(0, 0.01, n_days), index=dates)
    df['index_ret'] = df.index.get_level_values('date').map(idx_ret)
    return df, idx_ret


def test_max_drawdown_not_all_nan():
    """P0-1 回归：max_drawdown 不得全列为 NaN（cumprod NaN 污染）。"""
    daily, idx_ret = _make_daily()
    factors = calculate_factors(daily, idx_ret)
    assert 'low_risk' in factors.columns
    # 有足够历史的截面，low_risk NaN 率必须 < 5%
    latest = factors.index.get_level_values('date').max()
    cross = factors.xs(latest, level='date')
    nan_rate = cross['low_risk'].isna().mean()
    assert nan_rate < 0.05, f'low_risk NaN 率 {nan_rate:.1%}，mdd 污染疑似复发'


def test_low_risk_cross_section_varies():
    """low_risk 在截面上必须有区分度（非全同值/全零）。"""
    daily, idx_ret = _make_daily()
    factors = calculate_factors(daily, idx_ret)
    latest = factors.index.get_level_values('date').max()
    cross = factors.xs(latest, level='date')['low_risk'].dropna()
    assert len(cross) > 0
    assert cross.std() > 0.01, 'low_risk 截面无区分度'
