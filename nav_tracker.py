#!/usr/bin/env python3
# /root/sav3/nav_tracker.py
"""纸面净值追踪 — ARC Top50 等权组合的日频净值记录与窗口收益计算

口径说明（务必与展示文案一致）:
- 组合: 每日选股输出的 Top50（composite_score 排名前 50），等权，日频再平衡
- 收益: 条目 D 的 ret = 上一记录日选出的 Top50 持有到 D 的等权收益
  （收盘价比价，停牌股按最后可得收盘价结转，贡献 0 收益）
- 不含交易成本/印花税/滑点；使用原始收盘价（未复权，除权日有噪声，与系统回测口径一致）
- 存储: output/nav_history.json，每日一条，同日重跑覆盖当日 codes（ret 不变）
"""

import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger("nav_tracker")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
NAV_FILE = os.path.join(OUTPUT_DIR, 'nav_history.json')

# 窗口收益所需的最小覆盖率（不足则显示"数据积累中"）
MIN_COVERAGE = 0.6
# 每月约 21 个交易日
TRADE_DAYS_PER_MONTH = 21


def load_history(nav_file: str = NAV_FILE) -> list:
    """加载净值历史，返回按日期升序的条目列表。"""
    if not os.path.exists(nav_file):
        return []
    try:
        with open(nav_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"读取 {nav_file} 失败: {e}")
        return []
    entries = data.get('entries', []) if isinstance(data, dict) else []
    entries = [e for e in entries if e.get('date')]
    entries.sort(key=lambda e: e['date'])
    return entries


def _save_history(entries: list, nav_file: str = NAV_FILE):
    os.makedirs(os.path.dirname(nav_file), exist_ok=True)
    with open(nav_file, 'w', encoding='utf-8') as f:
        json.dump({'version': 1, 'entries': entries}, f, ensure_ascii=False, indent=1)


def _portfolio_return(codes: list, prev_date: pd.Timestamp, eff_date: pd.Timestamp,
                      close_pivot: pd.DataFrame):
    """上一记录日 Top50 持有到 eff_date 的等权收益（收盘价已 ffill，停牌结转）。"""
    valid = [c for c in codes if c in close_pivot.columns]
    if not valid:
        return None
    sub = close_pivot.loc[:eff_date, valid]
    if sub.empty:
        return None
    p0 = sub.loc[:prev_date].iloc[-1] if len(sub.loc[:prev_date]) else None
    p1 = sub.iloc[-1]
    if p0 is None:
        return None
    ret = (p1 / p0 - 1).replace([np.inf, -np.inf], np.nan).dropna()
    ret = ret[(ret > -1) & (ret < 2)]  # 防脏数据（单日 ±100%/200% 以外视为错误）
    if len(ret) < 10:
        logger.warning(f"净值收益样本不足: {len(ret)}/{len(valid)} 只有效定价")
        return None
    return float(ret.mean())


def update_nav_history(codes: list, daily_df: pd.DataFrame,
                       nav_file: str = NAV_FILE) -> dict:
    """每日选股后追加一条净值记录。

    codes: 当日 Top50 代码列表（ranked.index）
    daily_df: MultiIndex(date, code) 日K，需含 close 列

    返回 {'date', 'ret', 'nav', 'n_codes'}；同日重跑覆盖 codes 不改 ret。
    """
    if daily_df is None or daily_df.empty or 'close' not in daily_df.columns:
        raise ValueError("daily_df 为空或缺少 close 列")

    eff_date = daily_df.index.get_level_values('date').max()
    eff_str = eff_date.strftime('%Y-%m-%d')
    close_pivot = daily_df['close'].unstack('code').sort_index().ffill()

    history = load_history(nav_file)

    if history and history[-1]['date'] == eff_str:
        # 同日重跑：只更新当日选股名单，ret/nav 保持首次计算值
        history[-1]['codes'] = list(codes)
        entry = history[-1]
    else:
        prev = history[-1] if history else None
        ret = None
        if prev:
            ret = _portfolio_return(prev.get('codes', []), pd.Timestamp(prev['date']),
                                    eff_date, close_pivot)
        prev_nav = float(prev['nav']) if prev and prev.get('nav') else 1.0
        nav = prev_nav * (1.0 + (ret or 0.0))
        entry = {'date': eff_str, 'codes': list(codes),
                 'ret': round(ret, 6) if ret is not None else None,
                 'nav': round(nav, 6)}
        history.append(entry)

    _save_history(history, nav_file)
    logger.info(f"纸面净值已更新: {eff_str} nav={entry['nav']:.4f} "
                f"ret={entry['ret'] if entry['ret'] is not None else '—'} (共 {len(history)} 条)")
    return {'date': eff_str, 'ret': entry['ret'], 'nav': entry['nav'],
            'n_codes': len(codes), 'n_entries': len(history)}


def _window_return_from_entries(entries: list, as_of: pd.Timestamp, months: int):
    """从净值条目计算近 months 个月的累计收益。

    返回 (return or None, coverage_days, expected_days)。
    覆盖率不足 MIN_COVERAGE 时返回 (None, coverage, expected)。
    """
    ref = as_of - pd.DateOffset(months=months)
    rets = [e['ret'] for e in entries
            if e.get('ret') is not None and ref < pd.Timestamp(e['date']) <= as_of]
    expected = TRADE_DAYS_PER_MONTH * months
    if len(rets) < expected * MIN_COVERAGE:
        return None, len(rets), expected
    return float(np.prod([1.0 + r for r in rets]) - 1.0), len(rets), expected


def _window_return_from_index(index_close: pd.Series, as_of: pd.Timestamp, months: int):
    """指数近 months 个月收益（asof 口径：ref 日或之前最后一个收盘）。"""
    if index_close is None or len(index_close) == 0:
        return None
    close = index_close.sort_index()
    close.index = pd.to_datetime(close.index)
    upto = close.loc[:as_of]
    if len(upto) < 2:
        return None
    ref = as_of - pd.DateOffset(months=months)
    base = close.loc[:ref]
    if base.empty:
        return None
    return float(upto.iloc[-1] / base.iloc[-1] - 1.0)


def get_nav_performance(index_close: pd.Series = None, months_list=(1, 3, 12),
                        nav_file: str = NAV_FILE) -> dict:
    """计算策略 vs 基准的近 1/3/12 个月收益（供日报头部展示）。

    返回 {'as_of', 'n_entries', 'strategy': {'1m': r|None, ...},
          'benchmark': {...}, 'coverage': {'1m': (n, expected), ...}}
    历史不足时对应窗口为 None（日报显示"积累中"）。
    """
    entries = load_history(nav_file)
    if not entries:
        return {}
    as_of = pd.Timestamp(entries[-1]['date'])

    perf = {'as_of': entries[-1]['date'], 'n_entries': len(entries),
            'strategy': {}, 'benchmark': {}, 'coverage': {}}
    for m in months_list:
        r, cov, exp = _window_return_from_entries(entries, as_of, m)
        perf['strategy'][f'{m}m'] = r
        perf['benchmark'][f'{m}m'] = _window_return_from_index(index_close, as_of, m)
        perf['coverage'][f'{m}m'] = (cov, exp)
    return perf


def format_nav_line(perf: dict) -> str:
    """单行文本摘要（日志/调试用）。"""
    if not perf:
        return "无净值历史"

    def _fmt(v):
        return f"{v:+.1%}" if v is not None else "—"

    parts = []
    for k in ['1m', '3m', '12m']:
        s = _fmt(perf['strategy'].get(k))
        b = _fmt(perf['benchmark'].get(k))
        parts.append(f"{k.upper()}: 策略{s}/基准{b}")
    return f"截至{perf['as_of']} " + " | ".join(parts)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    p = get_nav_performance()
    print(format_nav_line(p))
    print(f"条目数: {p.get('n_entries', 0)}")
