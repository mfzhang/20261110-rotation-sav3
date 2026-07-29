#!/usr/bin/env python3
"""TID 行业轮动评估 — 默认生产口径，与 sector_rotation.py 完全一致

生产口径: 全市场（~5400股） × 申万29 × 成交额加权 × 20日夏普（ret_20d/vol_20d），
         月末调仓，次月持有至下一月末。

用法:
  python3 eval_tid.py                          # 生产口径，近 12 个月
  python3 eval_tid.py --start 2024-07-01 --end 2026-07-01
  python3 eval_tid.py --legacy                 # 旧口径（300池/简单平均/20日动量，仅对比用）
  python3 eval_tid.py --grid-delta             # 新闻 δ 网格回测（代理情绪，见下）
  python3 eval_tid.py --grid-delta --update-config   # 并把最优 δ 写入 config.py

δ 网格回测的数据说明（重要）:
  真实历史新闻情绪不可得（tushare news 接口无权限；6 个爬虫源只给当日头条）。
  本回测用「价格跳变代理情绪」：行业成交额加权收益的近3日异常度 z = r_3d/σ_60d，
  z 显著（|z|>0.5）视为该行业有新闻冲击，S=tanh(z/2)、conf=min(|z|/2,1)，
  再原样走生产的事件调节公式 M'=信号×(1+δ·S·conf)（复用 _finalize_industry_result）。
  新闻多数解释当日价格异动，该代理与生产情绪同向但不等同；结论用于 δ 量级选择，
  不用于精确胜率。真实情绪自 2026-07-19 起每日落盘 output/news_sent_history.jsonl，
  积累约 6 个月后可跑真实 δ 回测。
"""
import argparse
import logging
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config import NEWS_DELTA, CACHE_DIR
from data_fetcher import _get_pro, _call_with_retry, get_trade_dates, get_stock_basic, fetch_indices_daily
from sub_sectors import TUSHARE_TO_SHENWAN
import sector_rotation

LOOKBACK_DAYS = 70        # 信号窗口外的历史余量（代理情绪 σ60 + 20日信号窗口）
SIGNAL_WINDOW = 20        # 生产口径: 20 日夏普
PROXY_Z_MIN = 0.5         # 代理情绪最小 |z|（不足视为无新闻冲击，对应生产 n_news<2 门禁）
GRID_DELTAS = [0.0, 0.05, 0.1, 0.15, 0.2]


# ──────────────────────────────────────────────
# 数据：全市场历史日K（增量磁盘缓存 + 完整性门禁）
# ──────────────────────────────────────────────

def fetch_full_market_history(start: str, end: str) -> pd.DataFrame:
    """拉取 [start, end] 全部交易日的全市场日K（ts_code/pct_chg/amount/close）。

    磁盘缓存 tid_eval_mkt_{start}_{end}.pkl；失败交易日占比 >2% 不写缓存、直接报错
    （宁缺勿错，与 data_fetcher 同门禁）。
    """
    cache_file = os.path.join(CACHE_DIR, f'tid_eval_mkt_{start}_{end}.pkl')
    if os.path.exists(cache_file):
        try:
            df = pd.read_pickle(cache_file)
            if df is not None and not df.empty:
                print(f"命中全市场历史缓存: {os.path.basename(cache_file)} ({len(df):,} 行)")
                return df
        except Exception:
            pass

    dates = get_trade_dates(start, end)
    print(f"拉取全市场日K: {start}~{end}, {len(dates)} 个交易日")
    pro = _get_pro()
    frames, failed = [], 0
    for i, d in enumerate(dates):
        df = _call_with_retry(lambda dd=d: pro.daily(trade_date=dd), desc=f'eval mkt {d}')
        if df is None or df.empty:
            failed += 1
            continue
        df = df[['ts_code', 'pct_chg', 'amount', 'close']].copy()
        df['date'] = pd.to_datetime(d)
        frames.append(df.rename(columns={'ts_code': 'code'}))
        if (i + 1) % 50 == 0:
            print(f"  进度 {i + 1}/{len(dates)} (失败 {failed})")

    if not frames:
        raise RuntimeError("全市场日K拉取全部失败")
    fail_rate = failed / len(dates)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['date', 'code'], keep='last')
    combined = combined.set_index(['date', 'code']).sort_index()
    if fail_rate > 0.02:
        raise RuntimeError(f"全市场日K失败率 {failed}/{len(dates)} ({fail_rate:.1%}) > 2%，不写缓存")
    combined.to_pickle(cache_file)
    print(f"全市场日K: {len(combined):,} 行, {combined.index.get_level_values('date').nunique()} 个交易日, "
          f"失败 {failed}，已写缓存")
    return combined


# ──────────────────────────────────────────────
# 生产口径评估
# ──────────────────────────────────────────────

def _build_industry_daily(mkt: pd.DataFrame):
    """全市场日K → 申万29 成交额加权行业日收益矩阵（date × industry，小数）。

    与 sector_rotation._build_from_full_market 同口径（stock_basic 静态行业映射，
    幸存者偏差与生产一致——评估目标是复现生产信号质量）。
    """
    basic = get_stock_basic()
    ind_map = basic.set_index('ts_code')['industry'].dropna()
    ind_map = ind_map[ind_map != '']

    flat = mkt.reset_index()
    ret_matrix = flat.pivot(index='date', columns='code', values='pct_chg')
    amt_matrix = flat.pivot(index='date', columns='code', values='amount')

    industry_groups = {}
    for code, ind in ind_map.items():
        ind_sw = TUSHARE_TO_SHENWAN.get(ind, '综合')
        industry_groups.setdefault(ind_sw, []).append(code)

    ind_daily = {}
    for ind, codes in industry_groups.items():
        s = sector_rotation.volume_weighted_return(ret_matrix, amt_matrix, codes,
                                                   min_codes=3, ret_in_percent=True)
        if len(s) >= SIGNAL_WINDOW:
            ind_daily[ind] = s
    df = pd.DataFrame(ind_daily).sort_index()
    print(f"行业日收益矩阵: {df.shape[0]} 交易日 × {df.shape[1]} 个申万行业")
    return df, industry_groups


def _proxy_sentiment_events(ind_daily: pd.DataFrame, i: int) -> pd.DataFrame:
    """价格跳变代理情绪 → events_sent DataFrame（生产事件调节器的输入格式）。

    对信号日 i：z = 近3日均收益 / 近60日收益σ；|z|>PROXY_Z_MIN 的行业生成 3 条
    伪事件（近3日各一），sentiment_raw=tanh(z/2)，confidence=min(|z|/2,1)。
    """
    dates = ind_daily.index
    win3 = ind_daily.iloc[max(0, i - 2):i + 1]
    win60 = ind_daily.iloc[max(0, i - 59):i + 1]
    r3 = win3.mean()
    sigma = win60.std().replace(0, np.nan)
    z = (r3 / sigma).dropna()

    rows = []
    for ind, zv in z.items():
        if abs(zv) <= PROXY_Z_MIN:
            continue
        s = float(np.tanh(zv / 2.0))
        conf = float(min(abs(zv) / 2.0, 1.0))
        for d in win3.index[-3:]:
            rows.append({'date': d, 'industry': ind, 'sentiment_raw': s, 'confidence': conf})
    return pd.DataFrame(rows)


def _month_ends(dates: pd.DatetimeIndex, start: str, end: str) -> list:
    """[start, end] 内每月最后一个交易日（在 dates 中的位置）。"""
    in_range = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if len(in_range) == 0:
        return []
    me = pd.Series(in_range).groupby(in_range.to_period('M')).last()
    return [dates.get_loc(d) for d in me.values]


def eval_production(mkt: pd.DataFrame, index_close: pd.Series,
                    start: str, end: str, delta: float = 0.0,
                    use_proxy_sentiment: bool = False) -> dict:
    """生产口径月度回放，返回每月明细 DataFrame + 汇总指标。"""
    ind_daily, industry_groups = _build_industry_daily(mkt)
    dates = ind_daily.index
    month_ends = _month_ends(dates, start, end)

    idx_close = index_close.sort_index()
    idx_close.index = pd.to_datetime(idx_close.index)

    rows = []
    for k in range(len(month_ends) - 1):
        i, j = month_ends[k], month_ends[k + 1]
        if i < SIGNAL_WINDOW:
            continue
        dt, nxt = dates[i], dates[j]

        # 信号窗口：复用生产 finalize（夏普主信号 + stocks>=5 过滤 + 事件调节器）
        sig_win = ind_daily.iloc[i - SIGNAL_WINDOW + 1:i + 1]
        industry_daily = {ind: sig_win[ind].dropna() for ind in sig_win.columns}
        events = _proxy_sentiment_events(ind_daily, i) if use_proxy_sentiment else None
        perf = sector_rotation._finalize_industry_result(
            industry_daily, industry_groups, SIGNAL_WINDOW,
            events_sent=events, delta=delta)
        if perf.empty or len(perf) < 10:
            continue

        # 前向窗口：(dt, 下一月末]
        fwd_win = ind_daily.iloc[i + 1:j + 1]
        if fwd_win.empty:
            continue
        fwd = (1 + fwd_win).prod() - 1

        common = perf['industry'][perf['industry'].isin(fwd.index)]
        perf = perf[perf['industry'].isin(common)].reset_index(drop=True)
        if len(perf) < 10:
            continue
        fwd = fwd.loc[common.values]

        top3 = perf.head(3)['industry'].values
        bot3 = perf.tail(3)['industry'].values
        top_ret = fwd.loc[top3].mean()
        bot_ret = fwd.loc[bot3].mean()

        idx_seg = idx_close.loc[(idx_close.index > dt) & (idx_close.index <= nxt)]
        idx_base = idx_close.loc[:dt]
        idx_ret = float(idx_seg.iloc[-1] / idx_base.iloc[-1] - 1.0) if len(idx_seg) and len(idx_base) else np.nan

        rank_ic = perf.set_index('industry')['sharpe'].corr(fwd, method='spearman')
        rows.append({
            'month': nxt.strftime('%Y-%m'),
            'top3': ','.join(top3), 'bot3': ','.join(bot3),
            'top_ret': top_ret, 'bot_ret': bot_ret,
            'excess_tb': top_ret - bot_ret,
            'excess_idx': top_ret - idx_ret,
            'idx_ret': idx_ret,
            'rank_ic': rank_ic,
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {'detail': detail, 'summary': None}
    summary = {
        'n': len(detail),
        'win_tb': float((detail['excess_tb'] > 0).mean()),
        'avg_excess_tb': float(detail['excess_tb'].mean()),
        'win_idx': float((detail['excess_idx'] > 0).mean()),
        'avg_excess_idx': float(detail['excess_idx'].mean()),
        'rank_ic': float(detail['rank_ic'].mean()),
    }
    return {'detail': detail, 'summary': summary}


def _print_summary(title: str, result: dict):
    s = result['summary']
    if not s:
        print(f"{title}: 无有效月份")
        return
    print(f"\n=== {title} ===")
    print(f"有效月份: {s['n']}")
    print(f"胜率 (Top3>Bottom3): {s['win_tb']:.1%}")
    print(f"月均超额 (T-B): {s['avg_excess_tb']:+.2%}")
    print(f"胜率 (Top3>沪深300): {s['win_idx']:.1%}")
    print(f"月均超额 (对沪深300): {s['avg_excess_idx']:+.2%}")
    print(f"RankIC 均值: {s['rank_ic']:+.3f}")


# ──────────────────────────────────────────────
# δ 网格回测
# ──────────────────────────────────────────────

def grid_delta(mkt: pd.DataFrame, index_close: pd.Series, start: str, end: str,
               deltas=GRID_DELTAS) -> pd.DataFrame:
    """对每个 δ 跑生产口径回放（价格跳变代理情绪），输出对比表。"""
    results = []
    for d in deltas:
        r = eval_production(mkt, index_close, start, end,
                            delta=d, use_proxy_sentiment=(d > 0))
        s = r['summary']
        if s:
            results.append({'delta': d, **s})
    tbl = pd.DataFrame(results)
    print("\n=== 新闻 δ 网格回测（生产口径 + 价格跳变代理情绪）===")
    print("⚠️ 代理情绪说明见文件头注释：非真实历史新闻，结论用于 δ 量级选择\n")
    if tbl.empty:
        print("无有效结果")
        return tbl
    show = tbl.copy()
    for c in ['win_tb', 'win_idx']:
        show[c] = (show[c] * 100).round(1).astype(str) + '%'
    for c in ['avg_excess_tb', 'avg_excess_idx']:
        show[c] = (show[c] * 100).round(2).astype(str) + '%'
    show['rank_ic'] = show['rank_ic'].round(3)
    print(show.to_string(index=False))
    return tbl


def pick_best_delta(tbl: pd.DataFrame) -> float:
    """最优 δ：月均超额(T-B)最大者优先；超额差异 <0.1pp 视为噪声级并列，
    按 RankIC 决胜（全排名质量，比头尾价差稳健）；再并列取较小 δ（越少干预越稳）。"""
    if tbl.empty:
        return NEWS_DELTA
    best_excess = tbl['avg_excess_tb'].max()
    tied = tbl[tbl['avg_excess_tb'] >= best_excess - 0.001]  # 0.1pp 内并列
    tied = tied.sort_values(['rank_ic', 'delta'], ascending=[False, True])
    return float(tied.iloc[0]['delta'])


def update_config_delta(new_delta: float, config_path: str = 'config.py'):
    """把 NEWS_DELTA 写入 config.py（正则就地替换，保留行内注释并追加溯源）。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        src = f.read()
    stamp = datetime.now().strftime('%Y-%m-%d')
    new_line = (f"NEWS_DELTA = {new_delta}  # δ网格回测 {stamp} 最优"
                f"（eval_tid.py --grid-delta，代理情绪口径）")
    out, n = re.subn(r'^NEWS_DELTA\s*=\s*[0-9.]+.*$', new_line, src,
                     count=1, flags=re.MULTILINE)
    if n == 0:
        raise RuntimeError("config.py 中未找到 NEWS_DELTA 行")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(out)
    print(f"\n✅ config.py NEWS_DELTA 已更新为 {new_delta}")


# ──────────────────────────────────────────────
# 旧口径（--legacy，仅对比用，保留原实现）
# ──────────────────────────────────────────────

def eval_legacy(start='2024-01-01', end='2026-01-01', limit=300):
    from backtest import run_arc_monthly_returns, prepare_backtest_inputs

    print("=== TID 评估（旧口径：300池/简单平均/20日动量，仅对比用）===")
    daily, idx_ret, industry_map, index_df, factors = prepare_backtest_inputs(
        start, end, limit, skip_financials=True)
    print(f"数据: {len(daily):,} 条")

    daily_flat = daily.reset_index()
    monthly = daily_flat.copy()
    monthly['year_month'] = monthly['date'].dt.to_period('M')
    month_ends = sorted(set(monthly.groupby('year_month')['date'].max().values))

    tid_signals = []
    for dt in month_ends[:-1]:
        mask = (monthly['date'] <= dt) & (monthly['date'] > dt - pd.Timedelta(days=40))
        window = monthly[mask]
        window['ind'] = window['code'].map(industry_map)
        ind_ret = window.groupby(['ind', 'date'])['pct_chg'].mean().reset_index()
        ind_cum = ind_ret.groupby('ind')['pct_chg'].sum().dropna()
        ind_cum = ind_cum[ind_cum.index != '']
        if len(ind_cum) < 10:
            continue
        ranked = ind_cum.sort_values(ascending=False)
        tid_signals.append({'date': dt, 'top3': ranked.head(3).index.tolist(),
                            'bottom3': ranked.tail(3).index.tolist()})
    print(f"TID 信号: {len(tid_signals)} 个月")

    wins, losses, excess_returns = 0, 0, []
    for sig in tid_signals:
        dt = sig['date']
        nxt_mask = (monthly['date'] >= dt + pd.Timedelta(days=1)) & \
                   (monthly['date'] <= dt + pd.Timedelta(days=36))
        nxt = monthly[nxt_mask]
        nxt['ind'] = nxt['code'].map(industry_map)
        top_ret = nxt[nxt['ind'].isin(sig['top3'])].groupby('ind')['pct_chg'].sum().mean()
        bot_ret = nxt[nxt['ind'].isin(sig['bottom3'])].groupby('ind')['pct_chg'].sum().mean()
        if pd.notna(top_ret) and pd.notna(bot_ret):
            excess_returns.append(top_ret - bot_ret)
            if top_ret - bot_ret > 0:
                wins += 1
            else:
                losses += 1

    win_rate = wins / (wins + losses) * 100 if (wins + losses) else 0
    avg_excess = np.mean(excess_returns) if excess_returns else 0
    print(f"\n=== 旧口径结果 ===")
    print(f"有效信号: {len(excess_returns)} 个月")
    print(f"胜率 (Top3 > Bottom3): {win_rate:.1f}% ({wins}/{wins + losses})")
    print(f"月均超额: {avg_excess:+.2f}%")

    print("\n计算 ARC 月度收益（旧口径相关性）...")
    arc_series = run_arc_monthly_returns(factors, daily, index_df, start, end,
                                         industry_map=industry_map)
    arc_aligned, tid_aligned = [], []
    for i, sig in enumerate(tid_signals):
        if i >= len(excess_returns):
            break
        nxt_start = sig['date'] + pd.Timedelta(days=1)
        nxt_end = nxt_start + pd.Timedelta(days=35)
        arc_in_month = arc_series[(arc_series.index >= nxt_start) & (arc_series.index <= nxt_end)]
        if len(arc_in_month) > 0:
            arc_aligned.append(arc_in_month.sum())
            tid_aligned.append(excess_returns[i])
    valid = [(a, t) for a, t in zip(arc_aligned, tid_aligned) if not np.isnan(t)]
    if len(valid) >= 3:
        arc_v, tid_v = zip(*valid)
        print(f"TID 超额 vs ARC 收益 相关性: {np.corrcoef(arc_v, tid_v)[0, 1]:+.3f}")
    else:
        print("数据不足，无法计算相关性")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def _fetch_hs300_close(start: str, end: str) -> pd.Series:
    idx = fetch_indices_daily(start, end, {'沪深300': '000300.SH'}, fields=['close'])
    return idx['沪深300']['close']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TID 行业轮动评估（生产口径）')
    parser.add_argument('--start', default=None, help='评估起点 YYYY-MM-DD（默认近12个月）')
    parser.add_argument('--end', default=None, help='评估终点 YYYY-MM-DD（默认今天）')
    parser.add_argument('--legacy', action='store_true', help='旧口径（仅对比用）')
    parser.add_argument('--grid-delta', action='store_true', help='新闻 δ 网格回测')
    parser.add_argument('--update-config', action='store_true',
                        help='把网格最优 δ 写入 config.py 的 NEWS_DELTA')
    args = parser.parse_args()

    # 月末回放会逐月调用生产 finalize，静默其 INFO 避免刷屏
    logging.getLogger('sector_rotation').setLevel(logging.WARNING)

    end = args.end or datetime.now().strftime('%Y-%m-%d')
    start = args.start or (pd.Timestamp(end) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')

    if args.legacy:
        eval_legacy(start=start, end=end)
        sys.exit(0)

    data_start = (pd.Timestamp(start) - pd.Timedelta(days=int(LOOKBACK_DAYS * 1.6))).strftime('%Y-%m-%d')
    mkt = fetch_full_market_history(data_start, end)
    hs300 = _fetch_hs300_close(data_start, end)

    if args.grid_delta:
        tbl = grid_delta(mkt, hs300, start, end)
        best = pick_best_delta(tbl)
        print(f"\n最优 δ = {best}（当前 config NEWS_DELTA = {NEWS_DELTA}）")
        if args.update_config:
            if abs(best - NEWS_DELTA) > 1e-9:
                update_config_delta(best)
            else:
                print(f"NEWS_DELTA 已是最优值 {NEWS_DELTA}，无需更新")
    else:
        print(f"=== TID 生产口径评估: {start} ~ {end}（月末调仓，全市场×申万29×成交额加权×20日夏普）===")
        result = eval_production(mkt, hs300, start, end, delta=0.0)
        _print_summary("TID 生产口径（δ=0 纯信号基线）", result)
        if not result['detail'].empty:
            print("\n月度明细:")
            d = result['detail'].copy()
            for c in ['top_ret', 'bot_ret', 'excess_tb', 'excess_idx', 'idx_ret']:
                d[c] = (d[c] * 100).round(2)
            d['rank_ic'] = d['rank_ic'].round(3)
            print(d.to_string(index=False))
