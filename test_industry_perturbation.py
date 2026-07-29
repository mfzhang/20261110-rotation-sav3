#!/usr/bin/env python3
"""行业分类敏感性分析 — 测试前视偏差对回测结果的影响

对 industry_map 随机扰动 5%/10% 的股票行业标签，每种水平跑 10 次独立回测，
对比年化收益、夏普比率、最大回撤的波动，评估前视偏差风险等级。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import io
import contextlib
import pandas as pd
import numpy as np

from config import *
from backtest import backtest_dual_mode, compute_sect_momentum_percentile, prepare_backtest_inputs, make_cstar_modifier
from data_fetcher import fetch_index_daily

# ===== 参数（与 run_cstar_backtest.py 保持一致）=====
LAMBDA = 0.12
ALPHA = 3.0
START = '2024-01-01'
END = '2026-01-01'
LIMIT = 300
N_RUNS = 10


# ===================================================================
# 扰动函数
# ===================================================================

def perturb_industry_map(industry_map, fraction=0.05, seed=None):
    """随机扰动 industry_map：选取 fraction 比例的股票，替换为随机行业。

    Args:
        industry_map: pd.Series, index=ts_code, values=industry
        fraction: 扰动比例 (0.0 ~ 1.0)
        seed: 随机种子

    Returns:
        pd.Series: 扰动后的 industry_map（新对象，不修改原对象）
    """
    rng = np.random.default_rng(seed)
    perturbed = industry_map.copy()
    codes = perturbed.index.tolist()
    n_perturb = max(1, int(len(codes) * fraction))
    selected = rng.choice(codes, size=n_perturb, replace=False)

    all_industries = perturbed.unique().tolist()
    for code in selected:
        original = perturbed[code]
        candidates = [ind for ind in all_industries if ind != original]
        if candidates:
            perturbed[code] = rng.choice(candidates)

    return perturbed


# ===================================================================
# 单次回测（静默模式）
# ===================================================================

def run_single_backtest(daily, factors, index_df, industry_map):
    """执行单次 C* 回测，返回 performance metrics dict（静默，不打印）。"""
    sect_pct = compute_sect_momentum_percentile(daily, industry_map)
    cstar, _ = make_cstar_modifier(sect_pct, lambda_=LAMBDA, alpha=ALPHA)

    with contextlib.redirect_stdout(io.StringIO()):
        result = backtest_dual_mode(
            factors, daily, index_df, START, END,
            industry_map=industry_map,
            score_modifier=cstar,
        )
    return result


# ===================================================================
# 主流程
# ===================================================================

if __name__ == '__main__':
    print(f"=== 行业分类敏感性分析 {START} ~ {END} ===")
    print(f"λ={LAMBDA}, α={ALPHA}, top_n={TOP_N}")
    print(f"扰动水平: 5%, 10%  各 {N_RUNS} 次独立回测")
    print()

    # ---- 1. 数据准备（只拉一次）----
    print(f"[1/5] 准备数据 ({(pd.to_datetime(START) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')} ~ {END})...")
    daily, idx_ret, industry_map, index_df, factors = prepare_backtest_inputs(START, END, LIMIT, skip_financials=True)
    print(f"      有效记录: {len(daily):,}")
    print(f"      行业映射: {len(industry_map)} 只股票, {industry_map.nunique()} 个行业")

    # ---- 2. 因子计算（只算一次，industry_map 不影响因子计算）----
    print(f"[2/5] 计算因子...")
    print(f"      因子维度: {factors.shape}")

    # ---- 3. 指数数据（已由 prepare_data 返回）----
    print(f"[3/5] 获取指数数据...")
    # index_df 已从 prepare_data 复用
    print(f"      指数记录: {len(index_df)}")

    # ---- 4. 原始（无扰动）回测 ----
    print(f"\n[4/5] 原始（无扰动）回测...")
    result_orig = run_single_backtest(daily, factors, index_df, industry_map)
    if result_orig is None:
        print("ERROR: 原始回测无结果，终止。")
        sys.exit(1)

    r = result_orig
    print(f"      年化收益: {r['annual_return']*100:.2f}%  "
          f"夏普: {r['sharpe']:.2f}  "
          f"最大回撤: {r['max_drawdown']*100:.2f}%")

    # ---- 5. 扰动回测 ----
    print(f"\n[5/5] 扰动回测...")
    results = {}

    for pct_level in [0.05, 0.10]:
        label = f"{int(pct_level*100)}%"
        metrics = {'annual_return': [], 'sharpe': [], 'max_drawdown': []}

        for run_i in range(1, N_RUNS + 1):
            print(f"      扰动 {label} 第 {run_i}/{N_RUNS} 次...", end=' ', flush=True)
            seed = run_i * 100 + int(pct_level * 1000)
            ind_pert = perturb_industry_map(industry_map, fraction=pct_level, seed=seed)
            result_p = run_single_backtest(daily, factors, index_df, ind_pert)

            if result_p is not None:
                metrics['annual_return'].append(result_p['annual_return'])
                metrics['sharpe'].append(result_p['sharpe'])
                metrics['max_drawdown'].append(result_p['max_drawdown'])
                print("✓")
            else:
                print("✗ 无结果")

        results[label] = metrics

    # ---- 6. 输出报告 ----
    print()
    print("=" * 60)
    print("           行业分类敏感性分析报告")
    print("=" * 60)
    print()

    print(f"原始（无扰动）:")
    print(f"  年化收益: {r['annual_return']*100:.2f}%, 夏普: {r['sharpe']:.2f}, 最大回撤: {r['max_drawdown']*100:.2f}%")
    print()

    for pct_level, label in [(0.05, '5%'), (0.10, '10%')]:
        m = results[label]
        n_valid = len(m['annual_return'])
        if n_valid == 0:
            print(f"{label} 扰动（{N_RUNS}次）: 无有效结果")
            print()
            continue

        arr = np.array(m['annual_return']) * 100
        sr = np.array(m['sharpe'])
        md = np.array(m['max_drawdown']) * 100

        print(f"{label} 扰动（{n_valid}/{N_RUNS}次）:")
        print(f"  年化收益: {arr.mean():.2f}% ± {arr.std():.2f}%（均值 ± 标准差）")
        print(f"  夏普:      {sr.mean():.2f} ± {sr.std():.2f}")
        print(f"  最大回撤:   {md.mean():.2f}% ± {md.std():.2f}%")
        print()

    # 结论：基于 5% 扰动水平的年化收益波动
    m5 = results.get('5%', {})
    ann_vals = np.array(m5.get('annual_return', [0])) * 100
    ann_std = ann_vals.std() if len(ann_vals) > 0 else 0

    if ann_std > 3:
        conclusion = "严重"
    elif ann_std > 1:
        conclusion = "需关注"
    else:
        conclusion = "可控"

    m10 = results.get('10%', {})
    ann_vals_10 = np.array(m10.get('annual_return', [0])) * 100
    ann_std_10 = ann_vals_10.std() if len(ann_vals_10) > 0 else 0

    print(f"结论: 前视偏差对回测的影响 [{conclusion}]")
    if ann_std > 1:
        print(f"（5%扰动下年化收益波动 {ann_std:.2f}% > 1% → 需关注）")
    if ann_std > 3:
        print(f"（5%扰动下年化收益波动 {ann_std:.2f}% > 3% → 严重）")
    if ann_std_10 > 0:
        print(f"（10%扰动下年化收益波动 {ann_std_10:.2f}%）")
    print()
