#!/usr/bin/env python3
"""动态 λ 行业动量修正回测：仅在趋势市（高离散度）激活 TID 调节"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from config import *
from backtest import (
    backtest_dual_mode,
    compute_market_dispersion,
    dynamic_lambda,
    compute_sect_momentum_percentile,
    prepare_backtest_inputs,
    make_dynamic_lambda_modifier,
)

# ===== 动态 λ 参数 =====
MAX_LAMBDA = 0.30    # 最大倾斜幅度（仅在 p≥0.7 时生效）
ALPHA = 3.0          # 敏感度（tanh 陡峭度）

START = '2024-01-01'
END = '2026-01-01'
LIMIT = 300


if __name__ == '__main__':
    print(f"=== 动态 λ 行业动量修正回测 {START} ~ {END} ===")
    print(f"max_λ={MAX_LAMBDA}, α={ALPHA}")
    print(f"规则: p<0.3→λ=0, 0.3≤p<0.7→λ=0, p≥0.7→λ=max_λ·(p-0.7)/0.3")
    print(f"权重: low_risk={FACTOR_WEIGHTS['low_risk']}, position={FACTOR_WEIGHTS['position']}")

    print(f"\n数据区间: {(pd.to_datetime(START) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')} ~ {END}")

    daily, idx_ret, industry_map, index_df, factors = prepare_backtest_inputs(START, END, LIMIT, skip_financials=True)
    print(f"有效记录: {len(daily):,}")
    print(f"因子维度: {factors.shape}")

    # 行业动量百分位（TID 修正的基础）
    print("计算行业动量百分位...")
    sect_pct = compute_sect_momentum_percentile(daily, industry_map)
    print(f"  有效覆盖: {sect_pct.notna().sum():,}/{len(sect_pct):,}")

    # 市场离散度（趋势市识别）
    print("计算市场离散度...")
    market_dispersion = compute_market_dispersion(daily, industry_map)
    print(f"  离散度序列长度: {len(market_dispersion)}")
    if len(market_dispersion) > 0:
        active_frac = (market_dispersion >= 0.7).mean()
        print(f"  高离散度(p≥0.7)占比: {active_frac*100:.1f}%")

    # === 动态 λ score_modifier ===
    dynamic_score_modifier, tilt_stats = make_dynamic_lambda_modifier(
        sect_pct, market_dispersion, max_lambda=MAX_LAMBDA, alpha=ALPHA
    )

    # 调用通用回测
    result = backtest_dual_mode(factors, daily, index_df, START, END,
                                industry_map=industry_map,
                                score_modifier=dynamic_score_modifier)

    if result:
        print(f"\n=== 动态 λ 回测结果 ===")
        print(f"年化收益: {result['annual_return']*100:.2f}%")
        print(f"夏普比率: {result['sharpe']:.2f}")
        print(f"最大回撤: {result['max_drawdown']*100:.2f}%")
        print(f"总收益:   {result['total_return']*100:.2f}%")
        print(f"总期数:   {result['n_periods']}  危机期: {result['crisis_periods']}")

        if tilt_stats:
            tilt_df = pd.DataFrame(tilt_stats).set_index('date')
            active_tilts = tilt_df[tilt_df['lambda'] > 0]
            print(f"\n=== 动态 λ 激活统计 ===")
            print(f"总调仓期: {len(tilt_df)}")
            print(f"λ>0 激活期: {len(active_tilts)} ({len(active_tilts)/len(tilt_df)*100:.1f}%)")
            print(f"非零 λ 均值: {active_tilts['lambda'].mean():.4f}")
            print(f"非零 λ 最大: {active_tilts['lambda'].max():.4f}")
            if len(active_tilts) > 0:
                print(f"非零期 k_mean: {active_tilts['k_mean'].mean():.4f}")
                print(f"非零期 k_std:  {active_tilts['k_std'].mean():.4f}")
            print(f"整体 k_mean: {tilt_df['k_mean'].mean():.4f}")
    else:
        print("回测无结果")
