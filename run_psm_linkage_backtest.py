#!/usr/bin/env python3
"""PSM 联动回测：纯 ARC vs ARC+PSM（三信号仓位 + 选股数量联动）

用法:
  python3 /tmp/sav3/run_psm_linkage_backtest.py

三个信号（简化版，从 PSM 缓存指数数据计算）：
  1. growth_above_ma60: 创业板 & 科创50 均站上 MA60
  2. scenario_3_prob > 0.45: 失败剧本主导（用 ret_5d 和 dist_20high 近似）
  3. 创业板 above_ma5
结果: exposure clamp [0.55, 1.0]，<0.75 时 top_n 从 20 缩到 15。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pickle
from config import *
from backtest import backtest_dual_mode, compute_performance_metrics, _prepare_backtest_data, prepare_backtest_inputs
from data_fetcher import fetch_index_daily

START, END, LIMIT = '2024-01-01', '2026-01-01', 200
CACHE_DIR = '/root/sav3_cache'

print(f"=== PSM 联动回测 {START} ~ {END} ===\n")

# ── 1. 加载 PSM 缓存指数数据 ──
idx_cache = {}
for key in ['000001.SH', '399006.SZ', '000688.SH']:
    fname = os.path.join('/tmp/sav3/cache', 
                         f"mt_idx_{key.replace('.','_')}_2015_2026.pkl")
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            df = pickle.load(f)
            idx_cache[key] = df
            print(f"  PSM 缓存: {key} {len(df)} 行")

cyb = idx_cache.get('399006.SZ')  # 创业板指
kc50 = idx_cache.get('000688.SH')  # 科创50

if cyb is None or kc50 is None:
    print("❌ PSM 缓存数据不全，请先运行 market_timing 自检拉取数据")
    sys.exit(1)

# ── 2. 统一数据管线（ARC 回测用） ──
daily, idx_ret, industry_map, index_df, factors = prepare_backtest_inputs(START, END, LIMIT, skip_financials=True)
print(f"\n  ARC 数据: {len(daily)} 行, {factors.index.get_level_values('code').nunique()} 只股票")

# ── 3. 构建 PSM 信号表（每个交易日一行） ──
def compute_psm_signals(cyb_df, kc50_df):
    """从日线数据计算三信号（无需完整 PSM pipeline）"""
    # 对齐到共同日期（科创50 起始较晚）
    common_dates = cyb_df.index.intersection(kc50_df.index)
    sig_cyb = cyb_df.loc[common_dates]
    sig_kc = kc50_df.loc[common_dates]
    
    signals = []
    
    for i in range(60, len(common_dates)):
        dt = common_dates[i]
        # 创业板
        cyb_close = float(sig_cyb['close'].iloc[i])
        cyb_ma60 = float(sig_cyb['close'].iloc[i-59:i+1].mean())
        cyb_ma5 = float(sig_cyb['close'].iloc[i-4:i+1].mean())
        cyb_ret_5d = cyb_close / float(sig_cyb['close'].iloc[i-5]) - 1
        cyb_20h = float(sig_cyb['close'].iloc[i-19:i+1].max())
        cyb_dist_20h = (cyb_close - cyb_20h) / cyb_20h
        
        # 科创50
        kc50_close = float(sig_kc['close'].iloc[i])
        kc50_ma60 = float(sig_kc['close'].iloc[i-59:i+1].mean())
        kc50_ret_5d = kc50_close / float(sig_kc['close'].iloc[i-5]) - 1
        
        # 信号
        growth_above_ma60 = cyb_close > cyb_ma60 and kc50_close > kc50_ma60
        cyb_above_ma5 = cyb_close > cyb_ma5
        
        # scenario_3_prob 近似：growth_strong_recovery=True 时 ~0.23，否则 ~0.30
        growth_strong = cyb_ret_5d > 0.05 and kc50_ret_5d > 0.05
        base_p3 = 0.23 if growth_strong else 0.30
        if not growth_above_ma60:
            base_p3 += 0.10
        # 距离20日高点越远，失败概率越高
        if cyb_dist_20h < -0.05:
            base_p3 += 0.05
        
        signals.append({
            'date': dt,
            'growth_above_ma60': growth_above_ma60,
            'cyb_above_ma5': cyb_above_ma5,
            'scenario_3_prob': min(base_p3, 0.60),
        })
    
    return pd.DataFrame(signals).set_index('date')

psm_signals = compute_psm_signals(cyb, kc50)
print(f"\n  PSM 信号表: {len(psm_signals)} 行")
print(f"  growth_above_ma60 占比: {psm_signals['growth_above_ma60'].mean():.1%}")
print(f"  cyb_above_ma5 占比: {psm_signals['cyb_above_ma5'].mean():.1%}")
print(f"  scenario_3 > 0.45 占比: {(psm_signals['scenario_3_prob'] > 0.45).mean():.1%}")

# ── 4. 预计算调仓日的 PSM 仓位系数 ──
factors_used, month_ends = _prepare_backtest_data(
    factors, daily, index_df, START, END, industry_map)

period_exposures = []
for dt in month_ends:
    if dt in psm_signals.index:
        s = psm_signals.loc[dt]
        exposure = 1.0
        if not s['growth_above_ma60']:
            exposure -= 0.10
        if s['scenario_3_prob'] > 0.45:
            exposure -= 0.15
        if not s['cyb_above_ma5']:
            exposure -= 0.10
        exposure = max(0.55, min(1.0, exposure))
        top_n = 20 if exposure >= 0.75 else 15
    else:
        exposure = 1.0
        top_n = 20
    period_exposures.append({'date': dt, 'exposure': exposure, 'top_n': top_n})

pe_df = pd.DataFrame(period_exposures)
print(f"\n  PSM 仓位分布:")
print(f"    exposure=1.0: {(pe_df['exposure']==1.0).sum()} 期")
print(f"    exposure<1.0: {(pe_df['exposure']<1.0).sum()} 期")
print(f"    top_n=15: {(pe_df['top_n']==15).sum()} 期")
if (pe_df['exposure']<1.0).sum() > 0:
    print(f"    低仓位期平均 exposure: {pe_df.loc[pe_df['exposure']<1.0, 'exposure'].mean():.2f}")

# ── 5. 调用扩展版 run_backtest_loop ──
from crisis_detector import CrisisDetector
from backtest import run_backtest_loop

factor_cols = [c for c in factors_used.columns]
normal_weights = pd.Series({f: FACTOR_WEIGHTS.get(f, 0) for f in factor_cols})
crisis_weights = pd.Series({f: CRISIS_WEIGHTS.get(f, 0) for f in factor_cols})


def psm_exposure_fn(score, f_slice, dt, current_mode, extra_info):
    """PSM 信号驱动的仓位调整：正常模式下按三信号缩放得分"""
    if current_mode != 'normal' or dt not in psm_signals.index:
        return score
    s = psm_signals.loc[dt]
    exposure = 1.0
    if not s['growth_above_ma60']:
        exposure -= 0.10
    if s['scenario_3_prob'] > 0.45:
        exposure -= 0.15
    if not s['cyb_above_ma5']:
        exposure -= 0.10
    exposure = max(0.55, min(1.0, exposure))
    return score * exposure if exposure < 1.0 else score


def psm_top_n_fn(dt, current_mode):
    """PSM 信号驱动的选股数量：正常模式下低 exposure 缩至 15"""
    if current_mode != 'normal':
        return TOP_N
    pe_row = pe_df[pe_df['date'] == dt]
    return int(pe_row['top_n'].iloc[0]) if len(pe_row) > 0 else TOP_N


def run_one_backtest(label, use_psm):
    """跑一次回测，返回 perf dict"""
    detector = CrisisDetector(fresh_start=True)
    exposure_fn = psm_exposure_fn if use_psm else None
    top_n_fn = psm_top_n_fn if use_psm else None

    returns_df, _, _ = run_backtest_loop(
        factors_used, daily, index_df, month_ends,
        normal_weights, crisis_weights, detector,
        exposure_fn=exposure_fn, top_n_fn=top_n_fn,
        with_cost=True,
    )
    return compute_performance_metrics(returns_df)

print(f"\n{'='*70}")
print("── ① 纯 ARC（Top 20，无 PSM）──")
r_arc = run_one_backtest("纯 ARC", use_psm=False)

print(f"\n── ② ARC+PSM（三信号仓位 + Top 20/15 切换）──")
r_psm = run_one_backtest("ARC+PSM", use_psm=True)

# ── 6. 汇总对比 ──
print(f"\n{'='*70}")
print(f"  方案        年化收益    夏普     最大回撤    总收益     PSM低仓位期")
print(f"{'-'*70}")
for name, r in [('纯 ARC', r_arc), ('ARC+PSM', r_psm)]:
    low_exp = pe_df[pe_df['exposure'] < 1.0] if name == 'ARC+PSM' else None
    low_n = len(low_exp) if low_exp is not None else 0
    if r:
        print(f"  {name:<10} {r['annual_return']*100:>7.2f}%  {r['sharpe']:>5.2f}  "
              f"{r['max_drawdown']*100:>7.2f}%  {r['total_return']*100:>7.2f}%  "
              f"{f'{low_n}期' if low_n>0 else '-'}")

diff_ann = (r_psm['annual_return'] - r_arc['annual_return']) * 100 if r_psm and r_arc else 0
diff_sh = r_psm['sharpe'] - r_arc['sharpe'] if r_psm and r_arc else 0
diff_mdd = (r_psm['max_drawdown'] - r_arc['max_drawdown']) * 100 if r_psm and r_arc else 0
print(f"\n  联动增量: 年化{diff_ann:+.2f}% | 夏普{diff_sh:+.2f} | MDD{diff_mdd:+.2f}%")

if diff_ann > 0:
    print("  ✅ PSM 联动提升了年化收益")
elif diff_ann > -0.5:
    print("  ⚠️ PSM 联动基本持平")
else:
    print("  ❌ PSM 联动降低了年化收益（类似 C*/动态λ 的负增量）")
