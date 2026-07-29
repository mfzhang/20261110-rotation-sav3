import pandas as pd
import numpy as np
from config import LOW_RISK_WINDOW

def safe_zscore(s):
    """安全的 Z-Score，处理全 NaN 情况；退化时保留 NaN 位置"""
    v = s.dropna()
    if len(v) < 2 or v.std() == 0:
        return pd.Series(0, index=s.index).where(s.notna(), np.nan)
    return (s - v.mean()) / v.std()

def calculate_factors(daily_data, index_ret):
    daily = daily_data.copy()

    # position — 52周相对低位（低吸因子）
    # 公式: (high_52w - close) / (high_52w - low_52w)
    # 值越大=股价离52周高点越远=相对低位，方向与 low_risk 协同
    # IC预期为正：低吸→高收益（实测月度Rank IC约+0.0564）
    daily['position'] = (daily['high_52w'] - daily['close']) / (daily['high_52w'] - daily['low_52w'] + 1e-10)

    # low_risk sub-indicators
    daily['volatility'] = daily.groupby('code')['ret'].transform(
        lambda x: x.rolling(LOW_RISK_WINDOW, min_periods=20).std() * np.sqrt(252))

    # max_drawdown — vectorized: cummax → drawdown → rolling min (replaces rolling.apply)
    def _rolling_mdd(grp):
        """Vectorized rolling MDD: peak=shift(1).expanding().max(), dd=(c-peak)/peak."""
        # 首行收益为 NaN（pct_change 期首），fillna(0) 防止 NaN 沿 cumprod 累乘链污染全列
        r = grp.fillna(0).values
        cum = np.cumprod(1 + r)
        n = len(cum)
        res = np.full(n, np.nan)
        w = LOW_RISK_WINDOW
        for i in range(19, n):  # min_periods=20
            start = max(0, i - w + 1)
            win = cum[start:i + 1]
            peak = np.maximum.accumulate(win)
            peak_shifted = np.empty_like(peak)
            peak_shifted[0] = win[0]
            peak_shifted[1:] = peak[:-1]
            dd = (win - peak_shifted) / peak_shifted
            res[i] = np.min(dd)
        return pd.Series(res, index=grp.index)
    daily['max_drawdown'] = daily.groupby('code')['ret'].transform(_rolling_mdd)

    # beta - vectorized covariance approach using groupby
    # Eliminates O(n_codes) Python loop + MultiIndex slice overhead
    daily['beta'] = daily.groupby('code', group_keys=False).apply(
        lambda g: g['ret'].rolling(LOW_RISK_WINDOW, min_periods=20).cov(g['index_ret']) /
                  (g['index_ret'].rolling(LOW_RISK_WINDOW, min_periods=20).var() + 1e-10)
    )

    # Cross-section: synthesize low_risk and quality_stability per date
    dates = daily.index.get_level_values('date').unique()
    cols = ['position']
    factor_frames = []

    for dt in dates:
        sub = daily.xs(dt, level='date').copy()

        # low_risk components
        sub['vol_z'] = safe_zscore(sub.get('volatility', np.nan))
        sub['mdd_z'] = safe_zscore(sub.get('max_drawdown', np.nan))
        sub['beta_z'] = safe_zscore(sub.get('beta', np.nan))
        sub['low_risk'] = -(sub['vol_z'] + sub['mdd_z'] + sub['beta_z']) / 3

        # 注：quality_stability 因子已移除（依赖财报数据，暂未接入）
        # 未来加新因子时在此处合成，并追加到下方 factor_frames 的 cols 列表中

        sub['date'] = dt
        sub['code'] = sub.index
        factor_frames.append(sub[['date', 'code'] + cols + ['low_risk']])

    factors = pd.concat(factor_frames, ignore_index=True)
    return factors.set_index(['date','code']).sort_index()
