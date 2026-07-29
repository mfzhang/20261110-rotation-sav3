#!/usr/bin/env python3
"""
test_ivol_ic.py
验证特质波动率 (idiosyncratic volatility, ivol) 因子的月度 Rank IC。

ivol 定义:
  - 对每只股票，用过去 252 个交易日的日收益做 OLS 回归:
      ret = α + β × index_ret + ε
  - ivol = std(ε) × √252 (年化特质波动率)
  - 预期 IC 方向: 负相关（低特质波动 → 高收益，"低波异象"）

用法:
  python3 -m py_compile test_ivol_ic.py && python3 -u test_ivol_ic.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from scipy import stats as sp_stats

from config import LOW_RISK_WINDOW
from backtest import prepare_backtest_inputs

# ══════════════════════════════════════════════════════════════════════════════
# 1. 数据准备
# ══════════════════════════════════════════════════════════════════════════════
# 用已有缓存: ts_daily_2022-01-01_2026-01-01_300_84c634ab3770.pkl
# start 往前推足够远以覆盖 252 日滚动窗口（最早 IC 从 2024-01-01 起算）
DATA_START = "2022-01-01"   # 命中现有缓存，避免重复拉数据
DATA_END   = "2026-01-01"
STOCK_LIMIT = 300
IC_START   = "2024-01-01"   # IC 验证区间起点
IC_END     = "2026-01-01"

print("=" * 65)
print("  特质波动率 (ivol) 因子 — 月度 Rank IC 验证")
print("=" * 65)
print(f"  数据区间: {DATA_START} ~ {DATA_END}")
print(f"  股票数量: {STOCK_LIMIT}")
print(f"  IC 区间:  {IC_START} ~ {IC_END}")
print()

# 加载数据（使用已有缓存）
print("[1/4] 加载数据...")
daily, index_ret, industry_map, _, factors = prepare_backtest_inputs(
    IC_START, DATA_END, STOCK_LIMIT, skip_financials=True
)
print(f"      daily 形状: {daily.shape}")
print(f"      股票数量:   {daily.index.get_level_values('code').nunique()}")
print(f"      日期范围:   {daily.index.get_level_values('date').min().date()} ~ "
      f"{daily.index.get_level_values('date').max().date()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. 计算 ivol 因子
# ══════════════════════════════════════════════════════════════════════════════
# 向量化滚动 OLS: 用公式 residual_var = Var(ret) - β² × Var(index_ret)
# 其中 β = Cov(ret, index_ret) / Var(index_ret)
# 推导: Var(ε) = Var(ret) - Cov(ret, idx)² / Var(idx) = Var(ret) - β² × Var(idx)

print("\n[2/4] 计算 ivol (滚动 252 日 OLS 残差标准差)...")

def compute_ivol(daily_df):
    """
    对每只股票计算滚动 252 日特质波动率。

    参数
    ----
    daily_df : DataFrame, MultiIndex(date, code), 需包含 'ret' 和 'index_ret' 列

    返回
    ----
    ivol : Series, MultiIndex(date, code), 年化特质波动率
    """
    n = LOW_RISK_WINDOW  # 252
    ivol_out = pd.Series(np.nan, index=daily_df.index, name='ivol')

    codes = daily_df.index.get_level_values('code').unique()
    n_codes = len(codes)

    for ci, code in enumerate(codes):
        mask = daily_df.index.get_level_values('code') == code
        g = daily_df.loc[mask]

        ret = g['ret']
        idx_ret = g['index_ret']

        # 滚动均值（min_periods=63: 允许早期窗口不足 252 日时仍产出值，
        # 至 2024-01-01 IC 起点时，各股已有 484 个交易日，实际窗口 = 252）
        min_p = 63
        roll_mean_ret  = ret.rolling(n, min_periods=min_p).mean()
        roll_mean_idx  = idx_ret.rolling(n, min_periods=min_p).mean()

        # 滚动方差 (ddof=1)
        roll_var_ret   = ret.rolling(n, min_periods=min_p).var(ddof=1)
        roll_var_idx   = idx_ret.rolling(n, min_periods=min_p).var(ddof=1)

        # 滚动协方差 (ddof=1, pandas 内置 — 正确处理有效窗口大小)
        roll_cov = ret.rolling(n, min_periods=min_p).cov(idx_ret)

        # β = Cov(ret, idx) / Var(idx)
        beta = roll_cov / (roll_var_idx + 1e-10)

        # 残差方差 = Var(ret) - β² × Var(idx)
        residual_var = roll_var_ret - beta**2 * roll_var_idx
        residual_var = residual_var.clip(lower=0.0)  # 数值稳定性

        # ivol = std(ε) × √252（年化）
        ivol = np.sqrt(residual_var) * np.sqrt(252)

        ivol_out.loc[mask] = ivol.values

        if (ci + 1) % 50 == 0:
            print(f"      进度: {ci+1}/{n_codes} 只股票")

    valid_frac = ivol_out.notna().mean()
    print(f"      ivol 有效值占比: {valid_frac:.1%}")
    return ivol_out

ivol_series = compute_ivol(daily)

# 将 ivol 加入 daily 便于后续截面提取
daily_with_ivol = daily.copy()
daily_with_ivol['ivol'] = ivol_series

# ══════════════════════════════════════════════════════════════════════════════
# 3. 获取 low_risk 因子（用于对比）
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3/4] 计算 low_risk 因子用于对比...")
print(f"      factors 形状: {factors.shape}")
print(f"      low_risk 有效值: {factors['low_risk'].notna().sum():,} / {len(factors):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. 月度 Rank IC 计算
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4/4] 计算月度 Rank IC...")

# 获取 IC 验证区间内的月末日期
all_dates = sorted(daily.index.get_level_values('date').unique())
all_dates_dt = pd.DatetimeIndex(all_dates)
all_dates_dt = all_dates_dt[(all_dates_dt >= IC_START) & (all_dates_dt <= IC_END)]
month_ends = sorted(
    pd.Series(all_dates_dt).groupby(all_dates_dt.to_period('M')).last().values
)
m0 = pd.Timestamp(month_ends[0])
m1 = pd.Timestamp(month_ends[-1])
print(f"      IC 区间月末数: {len(month_ends)} "
      f"({m0.strftime('%Y-%m-%d')} ~ {m1.strftime('%Y-%m-%d')})")


def compute_monthly_ic(factor_values, daily_df, month_ends_list,
                       min_stocks=10, label='factor'):
    """
    计算月度 Rank IC (Spearman 秩相关)。

    参数
    ----
    factor_values : Series, MultiIndex(date, code)
        因子值序列
    daily_df : DataFrame, MultiIndex(date, code)
        需包含 'close' 列，用于计算 forward 1-month return
    month_ends_list : list of Timestamp
        月末日期列表
    min_stocks : int
        最少共同股票数
    label : str
        因子名称（用于日志）

    返回
    ----
    ic_list : list of float
        每期 IC 值
    """
    ic_list = []
    skipped_short = 0
    skipped_error = 0

    for i in range(len(month_ends_list) - 1):
        dt, nxt = month_ends_list[i], month_ends_list[i + 1]

        try:
            f_slice = factor_values.xs(dt, level='date')
            cs = daily_df.xs(dt, level='date')['close']
            ce = daily_df.xs(nxt, level='date')['close']
        except KeyError:
            skipped_error += 1
            continue

        fwd_ret = ce / cs - 1.0

        f_clean = f_slice.dropna()
        fwd_clean = fwd_ret.dropna()
        common = f_clean.index.intersection(fwd_clean.index)

        if len(common) < min_stocks:
            skipped_short += 1
            continue

        try:
            ic, _ = sp_stats.spearmanr(
                f_clean.loc[common], fwd_clean.loc[common]
            )
            ic_list.append(ic)
        except Exception:
            skipped_error += 1
            continue

    if skipped_short or skipped_error:
        print(f"      [{label}] 跳过: 样本不足={skipped_short}, 异常={skipped_error}")

    return ic_list


# 计算 IC
ivol_ic_list  = compute_monthly_ic(
    daily_with_ivol['ivol'], daily_with_ivol, month_ends, label='ivol'
)
lr_ic_list    = compute_monthly_ic(
    factors['low_risk'], daily_with_ivol, month_ends, label='low_risk'
)


def report_ic(name, ic_list, expected_sign, check_significance=True):
    """格式化输出 IC 统计并返回 (mean, t, n, passed)"""
    if not ic_list:
        print(f"\n{'─'*55}")
        print(f"  {name}: 无有效 IC 数据")
        return None

    ic_s = pd.Series(ic_list)
    ic_mean = ic_s.mean()
    ic_std  = ic_s.std(ddof=1)
    n       = len(ic_list)
    t_stat  = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0
    ir      = ic_mean / (ic_std + 1e-10)

    # 方向
    if expected_sign < 0:
        dir_label = "负相关（低因子→高收益）"
        dir_ok = ic_mean < 0
    else:
        dir_label = "正相关（高因子→高收益）"
        dir_ok = ic_mean > 0

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  IC Mean:     {ic_mean:+.4f}")
    print(f"  IC Std:      {ic_std:.4f}")
    print(f"  t-stat:      {t_stat:+.2f}")
    print(f"  IR (IC/SD):  {ir:+.2f}")
    print(f"  n_periods:   {n}")
    print(f"  预期方向:     {dir_label}")

    if dir_ok:
        print(f"  ✅ 方向符合预期")
    else:
        print(f"  ⚠️  方向与预期不符（IC={ic_mean:+.4f}, 预期 {'负' if expected_sign<0 else '正'}）")

    # 显著性判断
    passed = True
    if check_significance:
        if abs(ic_mean) < 0.02 or abs(t_stat) < 2:
            print(f"\n  ⚠️  {name.split()[0]} 不显著（|IC|<0.02 或 |t|<2），不建议加入")
            passed = False
        else:
            print(f"\n  ✅ {name.split()[0]} 显著且方向正确，建议加入因子模型")

    return {'name': name, 'mean': ic_mean, 'std': ic_std, 't': t_stat,
            'ir': ir, 'n': n, 'ic_list': ic_list, 'passed': passed}


# 输出 ivol 结果
ivol_r = report_ic("ivol (特质波动率)", ivol_ic_list, expected_sign=-1)

# 输出 low_risk 结果
lr_r = report_ic("low_risk (低风险)", lr_ic_list, expected_sign=+1)

# ══════════════════════════════════════════════════════════════════════════════
# 5. 对比分析
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*55}")
print("  ivol vs low_risk 对比分析")
print(f"{'='*55}")

if ivol_ic_list and lr_ic_list:
    # 对齐到相同期数（取较短的）
    n_common = min(len(ivol_ic_list), len(lr_ic_list))
    ivol_aligned = pd.Series(ivol_ic_list[:n_common])
    lr_aligned   = pd.Series(lr_ic_list[:n_common])

    # IC 序列相关性
    ic_corr = ivol_aligned.corr(lr_aligned)
    print(f"  IC 序列相关性:         {ic_corr:+.4f}")

    # 均值差检验（配对 t-test）
    diff = ivol_aligned - lr_aligned
    diff_mean = diff.mean()
    diff_t = diff_mean / (diff.std(ddof=1) / np.sqrt(n_common)) if diff.std() > 0 else 0
    print(f"  IC 均值差 (ivol-low_risk): {diff_mean:+.4f}  (t={diff_t:+.2f})")

    # low_risk 含 vol 成分，ivol 是去 market 后的纯净版
    print()
    if abs(ic_corr) > 0.7:
        print(f"  ⚠️  ivol 与 low_risk 的 IC 序列高度相关 ({ic_corr:+.2f})")
        print(f"      两者可能捕捉相似的风险溢价，加入时注意共线性")
    else:
        print(f"  ℹ️  ivol 与 low_risk IC 相关性适中 ({ic_corr:+.2f})")

    if abs(diff_mean) < 0.005:
        print(f"  ℹ️  ivol 与 low_risk IC 均值接近（差={diff_mean:+.4f}），可能有替代关系")
    elif abs(diff_t) > 2:
        direction = "更强" if abs(ivol_r['mean']) > abs(lr_r['mean']) else "更弱"
        print(f"  ℹ️  ivol 的 IC {direction}（差={diff_mean:+.4f}, t={diff_t:+.2f}），二者存在差异")

    # 区间内 ivol 对 low_risk 的截面相关性
    print(f"\n  ── 截面相关性（每月末 ivol vs low_risk）──")
    x_corrs = []
    for dt in month_ends:
        try:
            iv = daily_with_ivol.xs(dt, level='date')['ivol'].dropna()
            lr = factors.xs(dt, level='date')['low_risk'].dropna()
            common = iv.index.intersection(lr.index)
            if len(common) >= 30:
                x_corrs.append(sp_stats.spearmanr(iv[common], lr[common])[0])
        except Exception:
            pass
    if x_corrs:
        xcorr_s = pd.Series(x_corrs)
        print(f"  截面 Spearman 相关: mean={xcorr_s.mean():+.4f}, "
              f"std={xcorr_s.std():.4f}, n={len(xcorr_s)}")
        if abs(xcorr_s.mean()) > 0.7:
            print(f"  ⚠️  ivol 与 low_risk 截面高度相关，信息重叠严重")
        elif abs(xcorr_s.mean()) > 0.4:
            print(f"  ℹ️  ivol 与 low_risk 截面中度相关，有一定增量信息")
        else:
            print(f"  ✅ ivol 与 low_risk 截面低相关，可独立提供增量")

# ══════════════════════════════════════════════════════════════════════════════
# 6. 最终结论
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*55}")
print("  最终结论")
print(f"{'='*55}")

if ivol_r:
    if ivol_r['passed']:
        print(f"  ✅ ivol 因子 IC={ivol_r['mean']:+.4f} t={ivol_r['t']:+.2f} — 有效，建议加入因子模型")
    else:
        print(f"  ⚠️  ivol 因子 IC={ivol_r['mean']:+.4f} t={ivol_r['t']:+.2f} — 不显著，不建议加入")
else:
    print(f"  ❌ ivol 因子无法计算有效 IC")

if lr_r:
    print(f"      low_risk IC={lr_r['mean']:+.4f} t={lr_r['t']:+.2f} (参考基准)")

print()
