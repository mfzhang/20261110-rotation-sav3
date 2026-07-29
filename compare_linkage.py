#!/usr/bin/env python3
"""ARC × TID 联动方式对比回测：纯ARC vs 固定C* vs 动态λ — 同数据同参数"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from config import *
from backtest import (
    backtest_dual_mode,
    run_backtest_loop,
    compute_performance_metrics,
    compute_market_dispersion,
    dynamic_lambda,
    compute_sect_momentum_percentile,
    prepare_backtest_inputs,
    make_cstar_modifier,
    make_dynamic_lambda_modifier,
)

START, END, LIMIT = '2024-01-01', '2026-01-01', 300
ALPHA = 3.0

print(f"=== ARC×TID 联动对比回测 {START} ~ {END} ===")
print(f"股票池: {LIMIT} 只 | α={ALPHA}\n")

# ── 1. 统一数据管线 ──
daily, idx_ret, industry_map, index_df, factors = prepare_backtest_inputs(START, END, LIMIT, skip_financials=True)

# 行业动量百分位
sect_pct = compute_sect_momentum_percentile(daily, industry_map)
print(f"行业动量覆盖: {sect_pct.notna().sum():,}/{len(sect_pct):,}")

# 市场离散度
market_dispersion = compute_market_dispersion(daily, industry_map)
active_frac = (market_dispersion >= 0.7).mean() if len(market_dispersion) > 0 else 0
print(f"离散度序列: {len(market_dispersion)} 天 | 高离散(p≥0.7): {active_frac*100:.1f}%\n")

results = {}

# ── 2. 纯 ARC（无任何修正） ──
print("── ① 纯 ARC ──")
r_arc = backtest_dual_mode(factors, daily, index_df, START, END,
                            industry_map=industry_map)
results['纯 ARC'] = r_arc

# ── 3. 固定 C*（λ=0.12 全年） ──
print("\n── ② 固定 C* (λ=0.12) ──")
LAMBDA_FIXED = 0.12
cstar_fixed, tilt_cstar = make_cstar_modifier(sect_pct, lambda_=LAMBDA_FIXED, alpha=ALPHA)

r_cstar = backtest_dual_mode(factors, daily, index_df, START, END,
                              industry_map=industry_map, score_modifier=cstar_fixed)
results['固定 C*'] = r_cstar

# ── 4. 动态 λ ──
print("\n── ③ 动态 λ (p≥0.7激活) ──")
MAX_LAMBDA = 0.30
dynamic_modifier, tilt_dyn = make_dynamic_lambda_modifier(
    sect_pct, market_dispersion, max_lambda=MAX_LAMBDA, alpha=ALPHA
)

r_dyn = backtest_dual_mode(factors, daily, index_df, START, END,
                            industry_map=industry_map, score_modifier=dynamic_modifier)
results['动态 λ'] = r_dyn

# ── 5. 汇总对比 ──
print(f"\n{'='*70}")
print(f"{'方案':<12} {'年化收益':>8} {'夏普':>6} {'最大回撤':>8} {'总收益':>8} {'期数':>5} {'危机':>4}")
print(f"{'-'*70}")
for name, r in results.items():
    if r:
        print(f"{name:<12} {r['annual_return']*100:>7.2f}% {r['sharpe']:>5.2f} "
              f"{r['max_drawdown']*100:>7.2f}% {r['total_return']*100:>7.2f}% "
              f"{r['n_periods']:>5} {r['crisis_periods']:>4}")
    else:
        print(f"{name:<12} {'无结果':>8}")

if tilt_dyn:
    td = pd.DataFrame(tilt_dyn).set_index('date')
    active = td[td['lambda'] > 0]
    print(f"\n动态λ激活: {len(active)}/{len(td)} 期 ({len(active)/len(td)*100:.0f}%) | "
          f"λ均值={active['lambda'].mean():.3f} | k均值={active['k_mean'].mean():.4f}")

if tilt_cstar:
    tc = pd.DataFrame(tilt_cstar).set_index('date')
    print(f"固定C*: k均值={tc['k_mean'].mean():.4f}")

# ── Position因子方向验证（月度Rank IC） ──
from scipy import stats as sp_stats

print(f"\n{'='*70}")
print("── Position因子方向验证（月度 Spearman Rank IC）──")
print("预期: IC>0（正相关 = 低吸 → 高收益，实测均值+0.0564）")

# 获取月频调仓日期
raw_dates = factors.index.get_level_values('date').unique()
dates_dt = pd.DatetimeIndex(sorted(raw_dates))
month_ends_ic = pd.Series(dates_dt).groupby(dates_dt.to_period('M')).last().values
month_ends_ic = sorted(month_ends_ic)

ic_list = []
for i in range(len(month_ends_ic) - 1):
    dt, nxt = month_ends_ic[i], month_ends_ic[i + 1]
    try:
        f_slice = factors.xs(dt, level='date')
        cs = daily.xs(dt, level='date')['close']
        ce = daily.xs(nxt, level='date')['close']
    except Exception:
        continue
    if 'position' not in f_slice.columns:
        continue
    pos = f_slice['position'].dropna()
    fwd_ret = ce / cs - 1
    common = pos.index.intersection(fwd_ret.dropna().index)
    if len(common) < 10:
        continue
    try:
        ic, _ = sp_stats.spearmanr(pos.loc[common], fwd_ret.loc[common])
        ic_list.append(ic)
    except Exception:
        continue

if ic_list:
    ic_series = pd.Series(ic_list)
    ic_mean = ic_series.mean()
    print(f"月度IC序列: mean={ic_mean:.4f}, std={ic_series.std():.4f}, "
          f"n={len(ic_series)}, IR={ic_mean/(ic_series.std()+1e-10):.2f}")
    if ic_mean < 0:
        print("⚠️ 警告: Position因子IC均值为负，与预期（正相关=低吸→高收益）不符！")
        print("   请检查因子公式: (high_52w-close)/(high_52w-low_52w)")
    else:
        print("✅ Position因子IC方向符合预期（正相关=低吸→高收益）。")
else:
    print("⚠️ 无法计算Position因子IC（有效交叉样本不足）")
