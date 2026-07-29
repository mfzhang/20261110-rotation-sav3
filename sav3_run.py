#!/usr/bin/env python3
"""
ARC 多因子选股/回测主程序
用法:
  选股:  python sav3_run.py --mode run --date 2026-06-17 --limit 50
  快速:  python sav3_run.py --mode run --date 2026-06-17 --quick
  回测:  python sav3_run.py --mode backtest --start 2024-01-01 --end 2026-06-17 --limit 100
"""
import argparse, sys, os
import pandas as pd
from config import *
from data_fetcher import get_stock_basic, fetch_index_daily
from data_preparation import prepare_data
from factor_calculator import calculate_factors
from factor_pipeline import pipeline_process
from score_ranker import compute_composite_score, rank_stocks
from backtest import backtest_dual_mode

def main():
    parser = argparse.ArgumentParser(description='ARC 多因子选股引擎')
    parser.add_argument('--mode', default='run', choices=['run','backtest'])
    parser.add_argument('--date', help='YYYY-MM-DD')
    parser.add_argument('--start'); parser.add_argument('--end')
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--quick', action='store_true', help='跳过财报')
    parser.add_argument('--output', default='output.csv')
    args = parser.parse_args()

    basic = get_stock_basic()
    name_map = basic.set_index('ts_code')['name']

    if args.mode == 'run':
        if not args.date: print("请指定 --date"); sys.exit(1)
        target = args.date
        start_data = (pd.to_datetime(target)-pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        print(f"\n=== ARC ===\n日期:{target} 数据:{start_data}~{target}\n")
        daily, idx_ret, ind, _ = prepare_data(start_data, target, stock_limit=args.limit, skip_financials=args.quick)
        print(f"数据:{len(daily):,}条\n计算因子...")
        factors = calculate_factors(daily, idx_ret)
        print("因子处理...")
        factors = pipeline_process(factors, ind)
        print("加权排名...")
        scores = compute_composite_score(factors, weights=FACTOR_WEIGHTS)
        result = rank_stocks(scores, date=pd.to_datetime(target))
        result['stock_name'] = result.index.map(name_map)
        result['industry'] = result.index.map(ind)
        out = result.reset_index().rename(columns={'code':'stock_code'})
        cols = ['stock_code','stock_name','industry']+FACTOR_ORDER+['composite_score','rank']
        out = out[[c for c in cols if c in out.columns]]
        out.to_csv(args.output, index=False)
        print(f"\n=== Top 20 ===\n{out.head(20)[['stock_code','stock_name','composite_score','rank']].to_string()}\n保存:{args.output}")

    elif args.mode == 'backtest':
        if not args.start or not args.end: print("请指定 --start --end"); sys.exit(1)
        s_data = (pd.to_datetime(args.start)-pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        print(f"\n=== 回测 ===\n区间:{args.start}~{args.end}")
        daily, idx_ret, ind, index_df = prepare_data(s_data, args.end, stock_limit=args.limit, skip_financials=args.quick)
        factors = calculate_factors(daily, idx_ret)
        factors = pipeline_process(factors, ind)
        scores = compute_composite_score(factors, weights=FACTOR_WEIGHTS)
        perf = backtest_dual_mode(scores, daily, index_df, args.start, args.end, industry_map=ind)
        if perf:
            for k,v in perf.items():
                if k not in ('cum_curve','returns_df'): print(f"  {k}:{v:.4f}")
        else:
            print("  回测无有效结果")

if __name__=='__main__': main()
