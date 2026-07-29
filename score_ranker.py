import pandas as pd
import numpy as np
from config import (FACTOR_WEIGHTS, CRISIS_WEIGHTS, DYNAMIC_FACTOR_WEIGHTS,
                    REGIME_CRASH_RET20, REGIME_CRASH_DD60, REGIME_TREND_RET20,
                    REGIME_CRASH_WEIGHTS, REGIME_TREND_WEIGHTS)


def compute_regime_weights(index_close, crisis_mode: str = 'normal',
                           enable_dynamic: bool = DYNAMIC_FACTOR_WEIGHTS) -> dict:
    """按市场状态动态配权（生产/回测共用，P1-6）。

    实证依据（review_2026-07-18 §3.1）：position 是崩盘/反转期权——
    崩盘月 RankIC +0.54，趋势市连续三个月 -0.14~-0.18。
    崩盘/超跌状态加权 position，趋势市降权 position，其余时间用基准权重。

    Args:
        index_close: 基准指数收盘价序列（截至当前的全历史）
        crisis_mode: 'crisis' 时直接返回危机防御权重
        enable_dynamic: False 时恒返回 FACTOR_WEIGHTS（A/B 对照用）

    Returns:
        (weights_dict, regime_label) — regime ∈ {'crash','trend','neutral','crisis'}
    """
    if crisis_mode == 'crisis':
        return dict(CRISIS_WEIGHTS), 'crisis'
    if not enable_dynamic:
        return dict(FACTOR_WEIGHTS), 'neutral'

    regime = 'neutral'
    weights = dict(FACTOR_WEIGHTS)
    try:
        close = pd.Series(index_close).dropna()
        if len(close) >= 60:
            ret20 = close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 else 0.0
            dd60 = close.iloc[-1] / close.iloc[-60:].max() - 1
            ma60 = close.iloc[-60:].mean()
            if ret20 <= REGIME_CRASH_RET20 or dd60 <= REGIME_CRASH_DD60:
                regime = 'crash'
                weights = dict(REGIME_CRASH_WEIGHTS)
            elif ret20 >= REGIME_TREND_RET20 and close.iloc[-1] > ma60:
                regime = 'trend'
                weights = dict(REGIME_TREND_WEIGHTS)
    except Exception:
        pass
    return weights, regime

def compute_composite_score(factors_df, weights=None):
    """计算综合得分（截面 rank 归一化后加权）。
    
    将每个因子在截面内映射到 [0,1] 分位（rank percentile），
    然后按权重加权得到 composite_score。
    这解决了 Z-score 等权相加时低变异因子被压缩的问题。
    
    weights 参数可选，传入则使用自定义权重（消除全局修改并发风险）。
    """
    wmap = weights if weights is not None else FACTOR_WEIGHTS

    def _rank_composite(group):
        """对单个日期的截面做 rank 归一化 + 加权"""
        group = group.copy()  # 先 copy 再改列，避免视图赋值污染
        composite = pd.Series(0.0, index=group.index)
        for fac, w in wmap.items():
            if fac in group.columns:
                # 截面 rank 分位，映射到 [0, 1]；缺失值填 0.5（中性）
                rank_pct = group[fac].rank(pct=True)
                composite += w * rank_pct.fillna(0.5)
                # 保存 rank 分位供日报展示
                group[f'{fac}_rank'] = rank_pct
        group['composite_score'] = composite
        return group

    result = factors_df.groupby(level='date', group_keys=False).apply(_rank_composite)
    return result

def compute_rolling_ic(factors_df, daily_df, n_periods=12, horizon=20):
    """计算因子滚动 Rank IC 和 IC_IR（默认 20 日前瞻，对齐月度调仓周期）。

    对每个因子，计算每期截面 Rank IC（因子值与前瞻 horizon 日收益的 Spearman 秩相关），
    取最近 n_periods 期的均值和 IR。

    Args:
        factors_df: MultiIndex (date, code)，含 low_risk, position 等因子列
        daily_df: MultiIndex (date, code)，含 'close' 列
        n_periods: 滚动窗口期数
        horizon: 前瞻收益窗口（交易日）。默认 20（≈月度调仓周期，review §3.4：
                 月度调仓策略看日度 IC 会高估因子有效性——次日反转噪声与
                 月度持有收益不是一回事）；horizon=1 仅用于日度 IC 对照实验

    Returns:
        dict: {factor: {'ic_mean': float, 'ic_ir': float, 'n_valid': int}}
    """
    factor_cols = ['low_risk', 'position']
    results = {}

    dates = sorted(factors_df.index.get_level_values('date').unique())
    closes = daily_df['close'].unstack('code').sort_index()
    daily_dates = closes.index

    for factor in factor_cols:
        if factor not in factors_df.columns:
            results[factor] = {'ic_mean': 0.0, 'ic_ir': 0.0, 'n_valid': 0}
            continue

        ic_list = []
        for i in range(len(dates) - 1):
            dt = dates[i]
            pos = daily_dates.searchsorted(dt)
            fwd_pos = pos + horizon
            if fwd_pos >= len(daily_dates):
                break
            fdt = daily_dates[fwd_pos]

            try:
                fv = factors_df.xs(dt, level='date')[factor].dropna()
                fwd_ret = (closes.loc[fdt] / closes.loc[dt] - 1).dropna()

                common = fv.index.intersection(fwd_ret.index)
                if len(common) < 30:
                    continue

                ic = fv.loc[common].rank().corr(
                    fwd_ret.loc[common].rank(), method='spearman')
                if not np.isnan(ic):
                    ic_list.append(ic)
            except Exception:
                continue

        if len(ic_list) > 0:
            ic_arr = np.array(ic_list)
            ic_mean = float(np.mean(ic_arr[-n_periods:]))
            ic_std = float(np.std(ic_arr[-n_periods:]))
            ic_ir = ic_mean / (ic_std + 1e-10) if ic_std > 0 else 0.0
            n_valid = min(len(ic_list), n_periods)
        else:
            ic_mean = 0.0
            ic_ir = 0.0
            n_valid = 0

        results[factor] = {'ic_mean': ic_mean, 'ic_ir': ic_ir, 'n_valid': len(ic_list)}

    return results


def rank_stocks(scores_df, date=None):
    if date:
        sub = scores_df.xs(date, level='date').copy()
        sub = sub.sort_values('composite_score', ascending=False)
        sub['rank'] = range(1, len(sub)+1)
        return sub
    return scores_df.groupby(level='date', group_keys=False).apply(
        lambda g: g.sort_values('composite_score', ascending=False).assign(rank=range(1, len(g)+1)))
