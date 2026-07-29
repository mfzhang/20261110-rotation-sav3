"""多日回测发酵信号的确认/失效条件命中率"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from fermentation import run_fermentation_pipeline
from fermentation_patterns import save_patterns, load_patterns, backtest_conditions
import logging
logging.basicConfig(level=logging.WARNING)

# 用缓存数据（不调 API）
print("加载缓存...")
# 找最新的 300 股缓存
import glob
caches = sorted(glob.glob('/tmp/sav3/cache/ts_daily_*300*.pkl'))
if not caches:
    print("❌ 无缓存，先跑一次 main.py 生成缓存")
    sys.exit(1)
cache_file = caches[-1]
print(f"  缓存: {cache_file}")

daily = pd.read_pickle(cache_file)
all_dates = sorted(daily.index.get_level_values('date').unique())
print(f"  交易日: {len(all_dates)} 天 ({all_dates[0].date()} ~ {all_dates[-1].date()})")

# 行业映射
from config import CACHE_DIR
basic_file = os.path.join(CACHE_DIR, 'ts_stock_basic.pkl')
if os.path.exists(basic_file):
    basic = pd.read_pickle(basic_file)
    industry_map = basic.set_index('ts_code')['industry']
else:
    print("❌ 无 stock_basic 缓存")
    sys.exit(1)

# 取最后 30 个交易日的月末（作为调仓日），跑发酵检测
month_ends = []
dates_dt = pd.DatetimeIndex(all_dates)
for m in pd.Series(dates_dt).groupby(dates_dt.to_period('M')):
    month_ends.append(m[1].iloc[-1])
month_ends = month_ends[-6:]  # 最近 6 个月末
print(f"  测试月末: {len(month_ends)} 个")

# 跑每个月末的发酵检测
for dt in month_ends:
    date_str = dt.strftime('%Y-%m-%d')
    # 限制 daily 到该日期之前
    daily_slice = daily.loc[daily.index.get_level_values('date') <= dt]
    result = run_fermentation_pipeline(daily_slice, industry_map)
    ferm_df = result['fermentation']
    if ferm_df.empty:
        print(f"  {date_str}: 无候选")
        continue
    
    # 用简单规则生成 LLM 替代（不需要真实 API）
    llm_fake = {}
    for _, row in ferm_df.iterrows():
        llm_fake[row['sub_sector']] = {
            'catalyst': f'{row["sub_sector"]}低位放量',
            'confirm_condition': f'{row["sub_sector"]}次日涨幅>1%',
            'fail_condition': f'{row["sub_sector"]}次日跌幅>1%',
        }
    save_patterns(ferm_df, llm_fake, date_str)
    print(f"  {date_str}: {len(ferm_df)} 候选 (A={len(ferm_df[ferm_df['tier']=='A'])}, B={len(ferm_df[ferm_df['tier']=='B'])}, C={len(ferm_df[ferm_df['tier']=='C'])})")

# 回测命中率
print("\n=== 回测确认/失效条件命中率 ===")
hit_result = backtest_conditions(days=90)
if hit_result.empty:
    print("❌ 无足够数据计算命中率（需要连续日期的缓存）")
else:
    print(hit_result.to_string())
    print(f"\n平均确认命中率: {hit_result['confirm_hit_rate'].mean():.1%}")
    print(f"平均失效命中率: {hit_result['fail_hit_rate'].mean():.1%}")
