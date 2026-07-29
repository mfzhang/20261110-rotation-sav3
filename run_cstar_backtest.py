#!/usr/bin/env python3
"""C* 条件因子增强回测：ARC 0.5/0.5 × 行业动量非线性修正"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from config import *
from backtest import (
    backtest_dual_mode,
    compute_sect_momentum_percentile,
    prepare_backtest_inputs,
    make_cstar_modifier,
)

# ===== C* 参数 =====
LAMBDA = 0.12   # 最大倾斜幅度（上浮/下浮不超过12%）
ALPHA = 3.0     # 敏感度（tanh 陡峭度）

START = '2024-01-01'
END = '2026-01-01'
LIMIT = 300


if __name__ == '__main__':
    print(f"=== C* 条件因子增强回测 {START} ~ {END} ===")
    print(f"λ={LAMBDA}, α={ALPHA}")
    print(f"权重: low_risk={FACTOR_WEIGHTS['low_risk']}, position={FACTOR_WEIGHTS['position']}")

    print(f"\n数据区间: {(pd.to_datetime(START) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')} ~ {END}")

    daily, idx_ret, industry_map, index_df, factors = prepare_backtest_inputs(START, END, LIMIT, skip_financials=True)
    print(f"有效记录: {len(daily):,}")
    print(f"因子维度: {factors.shape}")

    # 行业动量百分位（C* 修正的基础）
    print("计算行业动量百分位...")
    sect_pct = compute_sect_momentum_percentile(daily, industry_map)
    print(f"  有效覆盖: {sect_pct.notna().sum():,}/{len(sect_pct):,}")

    # === C* score_modifier（注入到通用回测循环）===
    cstar_modifier, tilt_stats = make_cstar_modifier(sect_pct, lambda_=LAMBDA, alpha=ALPHA)

    # 调用通用回测
    result = backtest_dual_mode(factors, daily, index_df, START, END,
                                industry_map=industry_map,
                                score_modifier=cstar_modifier)

    if result:
        print(f"\n=== C* 回测结果 ===")
        print(f"年化收益: {result['annual_return']*100:.2f}%")
        print(f"夏普比率: {result['sharpe']:.2f}")
        print(f"最大回撤: {result['max_drawdown']*100:.2f}%")
        print(f"总收益:   {result['total_return']*100:.2f}%")
        print(f"总期数:   {result['n_periods']}  危机期: {result['crisis_periods']}")

        if tilt_stats:
            tilt_df = pd.DataFrame(tilt_stats).set_index('date')
            print(f"行业倾斜: 均值 k={tilt_df['k_mean'].mean():.4f}  "
                  f"最大 k={tilt_df['k_mean'].max():.4f}")
    else:
        print("回测无结果")
