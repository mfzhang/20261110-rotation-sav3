"""全量回测：5000 只股票 2 年，发酵信号命中率 — 单批版"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import time
import logging
logging.basicConfig(level=logging.WARNING)

from fermentation import run_fermentation_pipeline
from fermentation_patterns import save_patterns, backtest_conditions
from config import CACHE_DIR

N_TOTAL = 5000

print("=" * 60)
print(f"发酵信号全量回测 ({N_TOTAL} 股)")
print("=" * 60)

t0 = time.time()

# 1. 基础数据
print("\n[1/4] 加载基础数据...")
basic_file = os.path.join(CACHE_DIR, 'ts_stock_basic.pkl')
basic = pd.read_pickle(basic_file)
industry_map = basic.set_index('ts_code')['industry']
all_codes_all = basic['ts_code'].tolist()
all_codes = all_codes_all[:min(len(all_codes_all), N_TOTAL)]
print(f"  全市场: {len(all_codes_all)} 只, 回测取 {len(all_codes)} 只")

# 2. 拉取日线
print(f"\n[2/4] 拉取日线...")
from data_fetcher import fetch_stock_daily
cache_pattern = os.path.join(CACHE_DIR, f'ts_daily_*_{len(all_codes)}_*.pkl')
caches = sorted(glob.glob(cache_pattern))
if caches:
    print(f"  复用缓存: {os.path.basename(caches[-1])}")
    daily = pd.read_pickle(caches[-1])
else:
    print(f"  无缓存，拉取 2 年日K...")
    daily = fetch_stock_daily(all_codes, '2024-06-01', '2026-06-20')

all_dates = sorted(daily.index.get_level_values('date').unique())
print(f"  交易日: {len(all_dates)} ({all_dates[0].date()} ~ {all_dates[-1].date()})")
print(f"  加载耗时: {time.time()-t0:.0f}s")

# 3. 月末
dates_dt = pd.DatetimeIndex(all_dates)
month_ends = []
for m in pd.Series(dates_dt).groupby(dates_dt.to_period('M')):
    month_ends.append(m[1].iloc[-1])
month_ends = [me for me in month_ends if me >= pd.Timestamp('2024-12-31')]
print(f"\n[3/4] 测试月末: {len(month_ends)} 个")

# 4. 发酵检测
print(f"\n[4/4] 发酵检测...")
ferm_count = 0
for i, dt in enumerate(month_ends):
    date_str = dt.strftime('%Y-%m-%d')
    daily_slice = daily.loc[daily.index.get_level_values('date') <= dt]

    try:
        result = run_fermentation_pipeline(daily_slice, industry_map)
    except Exception:
        continue

    ferm_df = result['fermentation']
    if ferm_df.empty:
        continue

    llm_fake = {}
    for _, row in ferm_df.iterrows():
        llm_fake[row['sub_sector']] = {
            'catalyst': f'{row["sub_sector"]}低位放量',
            'confirm_condition': f'{row["sub_sector"]}次日涨幅>1%',
            'fail_condition': f'{row["sub_sector"]}次日跌幅>1%',
        }
    save_patterns(ferm_df, llm_fake, date_str)
    ferm_count += 1

    if ferm_count % 5 == 0 or i == len(month_ends) - 1:
        a_cnt = len(ferm_df[ferm_df['tier'] == 'A'])
        b_cnt = len(ferm_df[ferm_df['tier'] == 'B'])
        c_cnt = len(ferm_df[ferm_df['tier'] == 'C'])
        print(f"  {date_str} [{i+1}/{len(month_ends)}]: {len(ferm_df)}候选 A={a_cnt} B={b_cnt} C={c_cnt}")

print(f"  共 {ferm_count} 个月末有候选")

# 5. 回测
print(f"\n回测命中率...")
hit_result = backtest_conditions(days=365)

if hit_result.empty:
    print("❌ 无足够数据")
else:
    hit_result = hit_result.sort_values('total_checks', ascending=False)
    print(f"\n{'='*80}")
    print(f"{'子方向':<16} {'确认命中':>8} {'失效命中':>8} {'确认次数':>8} {'失效次数':>8} {'总检查':>6}")
    print(f"{'='*80}")
    for _, row in hit_result.iterrows():
        print(f"{row['sub_sector']:<16} {row['confirm_hit_rate']:>7.0%} {row['fail_hit_rate']:>7.0%} "
              f"{int(row['confirm_total']):>8} {int(row['fail_total']):>8} {int(row['total_checks']):>6}")

    mean_confirm = hit_result['confirm_hit_rate'].mean()
    mean_fail = hit_result['fail_hit_rate'].mean()
    total = hit_result['total_checks'].sum()
    print(f"{'='*80}")
    print(f"  平均确认命中: {mean_confirm:.0%}  |  平均失效命中: {mean_fail:.0%}  |  总检查: {int(total)} 次")
    print(f"  子方向数: {len(hit_result)}")

print(f"\n总耗时: {time.time()-t0:.0f}s")
