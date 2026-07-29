#!/usr/bin/env python3
"""固定权重回测：ARC 0.5/0.5 + 危机防御"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from config import *
from backtest import backtest_dual_mode, prepare_backtest_inputs

START = '2024-01-01'
END = '2026-01-01'
LIMIT = 300

print(f"=== 固定权重回测 {START} ~ {END} ===")
print(f"股票池: {LIMIT} 只, 权重: low_risk={FACTOR_WEIGHTS['low_risk']}, position={FACTOR_WEIGHTS['position']}")
print(f"危机: 触发={CRISIS_IDX_DRAWDOWN}/{CRISIS_VOL_RATIO}x, 退出={CRISIS_EXIT_DRAWDOWN}/{CRISIS_EXIT_DAYS}d")

print(f"\n数据区间: {(pd.to_datetime(START) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')} ~ {END}")

daily, idx_ret, ind, index_df, factors = prepare_backtest_inputs(START, END, LIMIT, skip_financials=True)
print(f"有效记录: {len(daily):,}")
print(f"因子维度: {factors.shape}")

# Crisis index data from prepare_data
# index_df 已从 prepare_data 返回

result = backtest_dual_mode(factors, daily, index_df, START, END, industry_map=ind)

if result:
    print(f"\n=== 最终结果 ===")
    print(f"年化收益: {result['annual_return']*100:.2f}%")
    print(f"夏普比率: {result['sharpe']:.2f}")
    print(f"最大回撤: {result['max_drawdown']*100:.2f}%")
    print(f"总收益:   {result['total_return']*100:.2f}%")
    print(f"总期数:   {result['n_periods']}")
    print(f"危机期:   {result['crisis_periods']}")
else:
    print("回测无结果")
