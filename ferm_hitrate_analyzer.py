#!/usr/bin/env python3
# /root/sav3/ferm_hitrate_analyzer.py
"""A/B/C 分档命中率分析器 — 发酵信号的事后验证

数据源: output/ferm_patterns.json（fermentation_patterns.save_patterns 每日落盘）。
命中定义: 信号日次一交易日，该子方向成交额加权收益 > 0（BACKTEST_RETURN_THRESHOLD），
与 fermentation_patterns.backtest_conditions 的确认命中口径一致（量比条件在
分档聚合层面不适用——C 档无 LLM 确认条件，这里用统一的"次日涨"标准，三档可比）。

用法:
  python3 ferm_hitrate_analyzer.py [--days 20]     # 打印摘要并落盘 latest.json
"""

import argparse
import json
import logging
import os

import pandas as pd

import fermentation_patterns as fpat
from fermentation_patterns import BACKTEST_RETURN_THRESHOLD

logger = logging.getLogger("ferm_hitrate_analyzer")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
LATEST_FILE = os.path.join(OUTPUT_DIR, 'ferm_hitrate_latest.json')

TIERS = ('A', 'B', 'C')


def analyze_tier_hit_rates(days: int = 20, daily_df=None) -> dict:
    """计算最近 N 天 A/B/C 各档的次日命中率。

    daily_df: MultiIndex(date, code) 日K（含 pct_chg/amount/name）；
              None 时回退到 fermentation_patterns 的日线缓存加载。
    返回 {'days', 'checked_dates', 'as_of', 'tiers': {tier: {n, hit_rate, avg_next_ret}}}
    数据不足时返回 {}。
    """
    patterns = fpat.load_patterns(days=days)
    if not patterns:
        logger.info("无发酵模式数据（ferm_patterns.json），命中率分析跳过")
        return {}

    if daily_df is None:
        daily_df = fpat._load_latest_daily_cache()
    if daily_df is None or daily_df.empty:
        logger.info("无日线数据，命中率分析跳过")
        return {}

    rets_by_tier = {t: [] for t in TIERS}
    checked_dates = 0
    as_of = None
    for date_str in sorted(patterns.keys()):
        records = patterns[date_str]
        next_date = fpat._get_next_trading_day(date_str, daily_df)
        if next_date is None:
            continue  # 最新一天尚无次日行情，自然跳过
        next_returns = fpat._compute_sub_daily_return(daily_df, next_date)
        if not next_returns:
            continue
        checked_dates += 1
        as_of = date_str
        for rec in records:
            tier = rec.get('tier', '')
            if tier not in rets_by_tier:
                continue
            r = next_returns.get(rec.get('sub_sector'))
            if r is not None:
                rets_by_tier[tier].append(float(r))

    tiers = {}
    for tier, rets in rets_by_tier.items():
        if not rets:
            continue
        s = pd.Series(rets)
        tiers[tier] = {
            'n': int(len(s)),
            'hit_rate': round(float((s > BACKTEST_RETURN_THRESHOLD).mean()), 4),
            'avg_next_ret': round(float(s.mean()), 6),
        }

    if not tiers:
        return {}
    result = {'days': days, 'checked_dates': checked_dates, 'as_of': as_of, 'tiers': tiers}
    logger.info(f"分档命中率: 检查 {checked_dates} 个信号日, "
                + ", ".join(f"{t}档 {v['hit_rate']:.0%}(n={v['n']})" for t, v in tiers.items()))
    return result


def save_latest(summary: dict, path: str = LATEST_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


def load_latest(path: str = LATEST_FILE) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def format_hitrate_line(summary: dict) -> str:
    """日报发酵区引用的一行摘要。"""
    if not summary or not summary.get('tiers'):
        return ""
    emoji = {'A': '🟢', 'B': '🟡', 'C': '⚪'}
    parts = []
    for tier in TIERS:
        v = summary['tiers'].get(tier)
        if not v:
            continue
        parts.append(f"{emoji[tier]}{tier}档 {v['hit_rate']:.0%}"
                     f"(n={v['n']}, 次日{v['avg_next_ret']:+.1%})")
    if not parts:
        return ""
    return (f"分档命中率（近{summary.get('checked_dates', 0)}个信号日，"
            f"次日子方向收益>0 视为命中）: " + " | ".join(parts))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description='A/B/C 分档命中率分析器')
    parser.add_argument('--days', type=int, default=20, help='回看最近 N 个信号日')
    args = parser.parse_args()

    summary = analyze_tier_hit_rates(days=args.days)
    if not summary:
        print("数据不足（需要 ferm_patterns.json 模式数据 + 日线缓存）")
    else:
        print(format_hitrate_line(summary))
        save_latest(summary)
        print(f"已落盘: {LATEST_FILE}")
