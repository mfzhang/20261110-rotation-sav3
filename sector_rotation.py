#!/usr/bin/env python3
# /tmp/sav3/sector_rotation.py
# 板块推荐引擎 — 可共享 ARC 的 daily 数据，避免重复拉取
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config import NEWS_DELTA
from data_fetcher import _get_pro, _call_with_retry, get_trade_dates, get_stock_basic

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("sector_rotation")

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')

# tushare 原生行业 → 申万一级行业（110→29）— 统一维护在 sub_sectors.py
from sub_sectors import TUSHARE_TO_SHENWAN


# ── 全市场日K缓存（TID 行业指数全市场口径，修复：原 300 股随机池抽样偏差）──
FULL_MARKET_CACHE = os.path.join(CACHE_DIR, 'ts_mkt_daily.pkl')
FULL_MARKET_KEEP_DATES = 60  # 缓存保留的最近交易日数（TID 只需 ~25 天，多留余量）


def fetch_full_market_daily(days: int = 30, end_date: str = None):
    """拉取全市场最近 days 个交易日的日K（pct_chg/amount/close），增量磁盘缓存。

    与 ARC 300 股随机池不同，这里按 trade_date 拉取全市场（~5400 只/日），
    供 TID 行业指数的全市场口径使用。全链路 _call_with_retry 节流 + 重试；
    失败交易日占比 >2% 时不写缓存（与 data_fetcher 同门禁）。

    返回 MultiIndex(date, code) DataFrame；拉取失败返回 None（由调用方降级）。
    """
    try:
        end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        start = (end - pd.Timedelta(days=int(days * 2.2) + 10)).strftime('%Y-%m-%d')
        want = get_trade_dates(start, end.strftime('%Y-%m-%d'))[-(days + 5):]
        if not want:
            return None

        cached = None
        if os.path.exists(FULL_MARKET_CACHE):
            try:
                cached = pd.read_pickle(FULL_MARKET_CACHE)
            except Exception:
                cached = None

        have = set()
        if cached is not None and not cached.empty:
            have = set(cached.index.get_level_values('date').strftime('%Y%m%d'))
        missing = [d for d in want if d not in have]

        if not missing:
            return cached

        pro = _get_pro()
        frames = []
        if cached is not None and not cached.empty:
            frames.append(cached.reset_index())
        fetched = 0
        failed = 0
        for d in missing:
            df = _call_with_retry(lambda dd=d: pro.daily(trade_date=dd),
                                  desc=f'mkt daily {d}')
            if df is None or df.empty:
                failed += 1
                continue
            df = df[['ts_code', 'pct_chg', 'amount', 'close']].copy()
            df['date'] = pd.to_datetime(d)
            frames.append(df.rename(columns={'ts_code': 'code'}))
            fetched += 1
        logger.info(f"全市场日K增量拉取: {fetched}/{len(missing)} 个交易日 (失败 {failed})")

        if not frames:
            return None

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'code'], keep='last')
        combined = combined.set_index(['date', 'code']).sort_index()
        # 只保留最近 N 个交易日，控制缓存体积
        keep_dates = sorted(combined.index.get_level_values('date').unique())[-FULL_MARKET_KEEP_DATES:]
        combined = combined.loc[pd.IndexSlice[keep_dates, :]]

        # 完整性门禁：失败率 >2% 不写缓存（沿用旧缓存，宁可缺不可错）
        if missing and failed / len(missing) > 0.02:
            logger.error(f"全市场日K失败率 {failed}/{len(missing)} ({failed / len(missing):.1%}) > 2%，不写缓存")
        else:
            combined.to_pickle(FULL_MARKET_CACHE)
        return combined
    except Exception as e:
        logger.warning(f"全市场日K拉取失败（降级共享股票池口径）: {e}")
        return None


def _build_from_full_market(mkt_df: pd.DataFrame, days: int,
                            events_sent=None, delta=NEWS_DELTA) -> pd.DataFrame:
    """从全市场日K构建申万一级行业指数（成交额加权）。"""
    try:
        basic = get_stock_basic()  # list_status='L' 全市场（带 7 天缓存）
    except Exception as e:
        logger.warning(f"stock_basic 不可用，无法构建全市场行业映射: {e}")
        return pd.DataFrame()
    ind_map = basic.set_index('ts_code')['industry'].dropna()
    ind_map = ind_map[ind_map != '']

    all_dates = sorted(mkt_df.index.get_level_values('date').unique())
    if len(all_dates) < days:
        logger.error(f"全市场数据不足: {len(all_dates)} < {days}")
        return pd.DataFrame()
    recent = all_dates[-days - 5:]
    daily = mkt_df.loc[pd.IndexSlice[recent, :]]

    industry_groups = {}
    for code, ind in ind_map.items():
        ind_sw = TUSHARE_TO_SHENWAN.get(ind, '综合')
        industry_groups.setdefault(ind_sw, []).append(code)

    daily_flat = daily.reset_index()
    ret_matrix = daily_flat.pivot(index='date', columns='code', values='pct_chg')
    amt_matrix = daily_flat.pivot(index='date', columns='code', values='amount')

    industry_daily = {}
    for ind, codes in industry_groups.items():
        daily_mean = volume_weighted_return(ret_matrix, amt_matrix, codes,
                                            min_codes=3, ret_in_percent=True)
        if len(daily_mean) >= days // 2:
            industry_daily[ind] = daily_mean

    logger.info(f"行业数: {len(industry_daily)} (全市场口径, {ret_matrix.shape[1]} 只股票)")
    return _finalize_industry_result(industry_daily, industry_groups, days, events_sent, delta)


def volume_weighted_return(ret_matrix: pd.DataFrame, amt_matrix: pd.DataFrame,
                           codes: list, min_codes: int = 3,
                           ret_in_percent: bool = True) -> pd.Series:
    """
    计算给定代码列表的成交额加权日收益序列。

    Args:
        ret_matrix: date × code 收益率矩阵
        amt_matrix: date × code 成交额矩阵
        codes: 需要聚合的股票代码列表
        min_codes: 最小成分股数量，不足返回空 Series
        ret_in_percent: ret_matrix 中的收益率是否为百分数（默认 True，会除以 100）

    Returns:
        pd.Series[date] 日收益（小数）
    """
    ind_codes = [c for c in codes if c in ret_matrix.columns and c in amt_matrix.columns]
    if len(ind_codes) < min_codes:
        return pd.Series(dtype=float)
    sub_ret = ret_matrix[ind_codes]
    if ret_in_percent:
        sub_ret = sub_ret / 100.0
    sub_amt = amt_matrix[ind_codes].clip(lower=0)
    weighted_sum = (sub_ret * sub_amt).sum(axis=1)
    amt_sum = sub_amt.sum(axis=1)
    daily_mean = (weighted_sum / amt_sum.replace(0, np.nan)).dropna()
    return daily_mean


def get_industry_performance(days: int = 20, daily_df=None, industry_map=None,
                             events_sent=None, delta: float = NEWS_DELTA,
                             use_full_market: bool = False) -> pd.DataFrame:
    """
    返回 DataFrame(行业, sharpe, return, stocks)，按 20 日夏普降序。

    数据路径优先级：
    1. use_full_market=True 时走全市场口径（修复：原 300 股随机池有抽样偏差，
       行业指数不能代表真实行业表现）；全市场拉取失败自动降级路径 2。
    2. 传入 daily_df + industry_map 时，从已有日K聚合（共享 ARC 数据管线，零额外 API 调用）。
    3. 否则走独立拉取路径（向后兼容）。

    daily_df: MultiIndex(date, code) DataFrame，需包含 pct_chg, amount 列
    industry_map: Series(code -> 行业名)
    """
    # === 路径 A0: 全市场口径 ===
    if use_full_market:
        mkt = fetch_full_market_daily(days + 10)
        if mkt is not None and not mkt.empty:
            result = _build_from_full_market(mkt, days, events_sent, delta)
            if not result.empty:
                return result
        logger.warning("全市场数据不可用，降级为共享股票池口径")

    # === 路径 A: 共享 ARC 数据 ===
    if daily_df is not None and industry_map is not None:
        return _build_from_daily_df(daily_df, industry_map, days, events_sent, delta)
    
    # === 路径 B: 独立拉取（向后兼容） ===
    pro = _get_pro()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 15)

    # 行业分类
    cf_basic = os.path.join(CACHE_DIR, 'ts_stock_basic.pkl')
    if os.path.exists(cf_basic) and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cf_basic))).days < 7:
        basic = pd.read_pickle(cf_basic)
    else:
        dfs = []
        for exchange in ['SSE', 'SZSE']:
            df = pro.stock_basic(exchange=exchange, list_status='L',
                                fields='ts_code,name,industry')
            dfs.append(df)
        basic = pd.concat(dfs, ignore_index=True)
        basic.to_pickle(cf_basic)

    basic = basic[basic['industry'].notna() & (basic['industry'] != '')]
    basic['industry_sw'] = basic['industry'].map(TUSHARE_TO_SHENWAN).fillna('综合')
    industry_groups = basic.groupby('industry_sw')['ts_code'].apply(list).to_dict()
    logger.info(f"行业数: {len(industry_groups)}, 总股票: {len(basic)}")

    # 拉取全市场日收益 + 成交额
    all_returns = {}
    all_amounts = {}
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    for dt in dates:
        trade_date = dt.strftime('%Y%m%d')
        try:
            df = pro.daily(trade_date=trade_date)
            if df is not None and not df.empty:
                df = df[df['ts_code'].isin(basic['ts_code'])]
                df['pct_chg'] = pd.to_numeric(df['pct_chg'], errors='coerce')
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                all_returns[trade_date] = df.set_index('ts_code')['pct_chg'].dropna()
                all_amounts[trade_date] = df.set_index('ts_code')['amount'].dropna()
        except Exception as e:
            logger.debug(f"  {trade_date}: {e}")

    return _compute_industry_returns(all_returns, all_amounts, industry_groups, days,
                                    events_sent, delta)


def _build_from_daily_df(daily_df: pd.DataFrame, industry_map: pd.Series, days: int,
                        events_sent=None, delta=NEWS_DELTA) -> pd.DataFrame:
    """从 ARC 共享的 daily DataFrame 构建行业指数"""
    # 取最近 days+5 个交易日
    all_dates = sorted(daily_df.index.get_level_values('date').unique())
    if len(all_dates) < days:
        logger.error(f"数据不足: {len(all_dates)} < {days}")
        return pd.DataFrame()
    
    recent_dates = all_dates[-days-5:]
    daily = daily_df.loc[pd.IndexSlice[recent_dates, :]]
    
    # 行业分组
    industry_groups = {}
    for code, ind in industry_map.items():
        if pd.isna(ind) or ind == '':
            continue
        ind_sw = TUSHARE_TO_SHENWAN.get(ind, '综合')
        industry_groups.setdefault(ind_sw, []).append(code)
    
    logger.info(f"行业数: {len(industry_groups)} (共享 ARC 数据)")
    
    # 从 daily 提取 pct_chg 和 amount 矩阵 (date × code)
    daily_flat = daily.reset_index()
    
    if 'pct_chg' not in daily_flat.columns or 'amount' not in daily_flat.columns:
        logger.error("daily_df 缺少 pct_chg 或 amount 列")
        return pd.DataFrame()
    
    ret_matrix = daily_flat.pivot(index='date', columns='code', values='pct_chg')
    amt_matrix = daily_flat.pivot(index='date', columns='code', values='amount')
    
    # 成交额加权行业日收益
    industry_daily = {}
    for ind, codes in industry_groups.items():
        daily_mean = volume_weighted_return(ret_matrix, amt_matrix, codes,
                                            min_codes=3, ret_in_percent=True)
        if len(daily_mean) >= days // 2:
            industry_daily[ind] = daily_mean
    
    return _finalize_industry_result(industry_daily, industry_groups, days, events_sent, delta)


def _compute_industry_returns(all_returns, all_amounts, industry_groups, days,
                             events_sent=None, delta=NEWS_DELTA) -> pd.DataFrame:
    """从 API 拉取的数据计算行业收益（向后兼容路径 B）"""
    if len(all_returns) < days // 2:
        logger.error(f"交易日数据不足: {len(all_returns)} < {days//2}")
        return pd.DataFrame()

    ret_matrix = pd.DataFrame(all_returns).T
    ret_matrix.index = pd.to_datetime(ret_matrix.index, format='%Y%m%d')
    ret_matrix.sort_index(inplace=True)
    ret_decimal = ret_matrix / 100.0

    amt_matrix = pd.DataFrame(all_amounts).T
    amt_matrix.index = pd.to_datetime(amt_matrix.index, format='%Y%m%d')
    amt_matrix.sort_index(inplace=True)

    industry_daily = {}
    for ind, codes in industry_groups.items():
        daily_mean = volume_weighted_return(ret_decimal, amt_matrix, codes,
                                            min_codes=3, ret_in_percent=False)
        if len(daily_mean) >= days // 2:
            industry_daily[ind] = daily_mean

    return _finalize_industry_result(industry_daily, industry_groups, days, events_sent, delta)


def _finalize_industry_result(industry_daily, industry_groups, days,
                              events_sent=None, delta=NEWS_DELTA) -> pd.DataFrame:
    """累乘 → 20日夏普主信号 → 排名 → 事件调节器 → 输出"""
    if not industry_daily:
        logger.error("未生成任何行业日收益")
        return pd.DataFrame()

    ind_df = pd.DataFrame(industry_daily)
    ind_df = ind_df.iloc[-days:]

    cum_ret = ((1 + ind_df).prod() - 1) * 100  # 旧 20 日动量（保留为对比参考）

    # 主信号：20 日夏普 = 20 日累计收益 / 20 日窗口日收益标准差
    # （实证：20 日夏普变体优于纯动量，见 review_2026-07-18 / eval_tid 复算）
    vol_20d = ind_df.std()
    sharpe_20d = (cum_ret / 100.0) / vol_20d.replace(0, np.nan)

    result = pd.DataFrame({
        'industry': cum_ret.index,
        'sharpe': sharpe_20d.values,
        'return': cum_ret.values,
        'stocks': [len(industry_groups.get(ind, [])) for ind in cum_ret.index]
    })

    # === 事件调节器 M' = 信号 × (1 + δ × S × confidence) ===
    if events_sent is not None and not events_sent.empty:
        # 取最近 3 日平均情绪（滞后校正：用调仓日前的事件）
        sent_recent = events_sent.groupby('industry').apply(
            lambda g: g.sort_values('date').tail(3)
        ).reset_index()
        # 窗口内新闻条数（用于最小样本保护）
        window_count = sent_recent.groupby('industry').size().reset_index(name='n_news')
        agg_sent = sent_recent.groupby('industry').agg(
            S=('sentiment_raw', 'mean'),
            conf=('confidence', 'mean')  # 均值避免单一高置信事件虚高
        ).reset_index()

        result = result.merge(agg_sent, on='industry', how='left')
        result['S'] = result['S'].fillna(0)
        result['conf'] = result['conf'].fillna(0)
        # 最小样本保护：窗口内新闻 < 2 条的行业不参与事件调节
        result = result.merge(window_count, on='industry', how='left')
        result['n_news'] = result['n_news'].fillna(0).astype(int)
        mask_few = result['n_news'] < 2
        result.loc[mask_few, 'S'] = 0.0
        result.loc[mask_few, 'conf'] = 0.0

        factor = 1 + delta * result['S'] * result['conf']
        # 约束：S<0 时不加剧负动量（只遏制追涨，不加剧杀跌）
        adj_mask = ~((result['S'] < 0) & (result['return'] < 0))
        result.loc[adj_mask, 'return'] = result.loc[adj_mask, 'return'] * factor[adj_mask]
        result.loc[adj_mask, 'sharpe'] = result.loc[adj_mask, 'sharpe'] * factor[adj_mask]

    result.sort_values('sharpe', ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    result = result[result['stocks'] >= 5]

    logger.info(f"板块轮动完成: {len(result)} 个行业（按20日夏普排序）")
    return result


def recommend(daily_df=None, industry_map=None, events_sent=None,
              use_full_market: bool = False) -> tuple:
    """返回 (top3列表, bottom3列表, 完整DataFrame)

    top3/bottom3 条目为 [行业名, 20日夏普]（主信号）；
    旧 20 日动量保留在 perf['return'] 列作对比参考。
    """
    perf = get_industry_performance(days=20, daily_df=daily_df, industry_map=industry_map,
                                    events_sent=events_sent, use_full_market=use_full_market)
    if perf.empty:
        return [], [], perf

    n = len(perf)
    show_n = min(3, n // 2) if n >= 3 else n
    top3 = perf.head(show_n)[['industry', 'sharpe']].values.tolist()
    bottom3 = perf.tail(show_n)[['industry', 'sharpe']].values.tolist()
    return top3, bottom3, perf


def get_stock_picks(daily_df, industry_map, top_n_industries=3, picks_per_industry=5,
                    events_sent=None, use_full_market: bool = False):
    """
    在 TID 强势行业（按20日夏普排名）中挑选短期动量最强的个股。
    返回 list of dict: {industry, code, ts_code, name, pct_chg, close, buy_price, stop_loss}

    注意：行业指数/排名用全市场口径（use_full_market=True 时），
    个股仍从共享股票池中挑选（该池已经过流动性/ST/上市时长过滤，是可交易池）。
    """
    if daily_df is None or industry_map is None:
        return []

    # 获取行业排名
    perf = get_industry_performance(days=20, daily_df=daily_df, industry_map=industry_map,
                                    events_sent=events_sent, use_full_market=use_full_market)
    if perf.empty:
        return []

    top_industries = perf.head(top_n_industries)['industry'].tolist()
    if not top_industries:
        return []

    # 最新交易日个股数据
    latest_date = daily_df.index.get_level_values('date').max()
    try:
        latest = daily_df.xs(latest_date, level='date').copy()
    except Exception:
        return []

    latest['industry'] = latest.index.map(industry_map).map(
        lambda x: TUSHARE_TO_SHENWAN.get(x, '综合'))
    latest = latest.dropna(subset=['industry'])

    # 计算短期动量（优先用 ret_20d 或 pct_chg）
    # 注意单位：ret_20d 是小数（0.074=+7.4%），pct_chg 列是百分数（7.4）
    if 'ret_20d' in latest.columns:
        mom_col = 'ret_20d'
        mom_scale = 100.0   # 小数 → 百分数（P0-4 修复：日报按百分数展示）
    elif 'pct_chg' in latest.columns:
        mom_col = 'pct_chg'
        mom_scale = 1.0     # 已是百分数
    else:
        return []

    # ── 计算每只股票的 ATR(14)，用于止损价 ──
    # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    # （原实现只有 high-low，跳空日波动被低估）
    atr_last = pd.Series(dtype=float)
    if 'high' in daily_df.columns and 'low' in daily_df.columns:
        prev_close = daily_df.groupby(level='code')['close'].shift(1)
        tr = pd.concat([
            daily_df['high'] - daily_df['low'],
            (daily_df['high'] - prev_close).abs(),
            (daily_df['low'] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_last = tr.groupby(level='code').apply(
            lambda g: g.rolling(14, min_periods=14).mean().iloc[-1]
        )

    picks = []
    for ind in top_industries:
        subset = latest[latest['industry'] == ind].copy()
        subset = subset.dropna(subset=[mom_col, 'close'])
        if subset.empty or len(subset) < 3:
            continue
        top_stocks = subset.nlargest(min(picks_per_industry, len(subset)), mom_col)
        for code, row in top_stocks.iterrows():
            close_raw = row.get('close')
            close_val = float(close_raw) if pd.notna(close_raw) else None

            # 买入价：动量策略按市价入场，不加折扣
            if close_val is not None:
                buy_price = round(close_val, 2)
            else:
                buy_price = None

            # 止损价：close - 1.5 * ATR(14)，不足14天回退为 close * 0.92
            stop_loss = None
            if close_val is not None:
                ts_code = str(code)
                atr_val = atr_last.get(ts_code)
                if pd.notna(atr_val):
                    stop_loss = round(close_val - 1.5 * float(atr_val), 2)
                else:
                    # 数据不足14天或无ATR，使用8%绝对值止损
                    stop_loss = round(close_val * 0.92, 2)

            # 动量统一为百分数；数值域 sanity check（P0-4：防单位错误复发）
            mom_pct = float(row.get(mom_col, 0)) * mom_scale
            if abs(mom_pct) > 80:
                logger.warning(f"TID 动量超域: {code} {mom_pct:+.1f}%（疑似数据错误），跳过")
                continue

            picks.append({
                'industry': ind,
                'code': code.split('.')[0] if '.' in str(code) else str(code),
                'ts_code': str(code),
                'pct_chg': round(mom_pct, 1),
                'close': round(float(close_raw) if pd.notna(close_raw) else 0, 2),
                'name': row.get('name', ''),
                'buy_price': buy_price,
                'stop_loss': stop_loss,
            })

    # 全列 |动量| < 1% 亮黄灯：疑似单位再次缩小 100 倍
    if picks and all(abs(p['pct_chg']) < 1.0 for p in picks):
        logger.warning("TID 动量全列 |值|<1%，疑似单位错误（应已是百分数）")

    return picks


if __name__ == '__main__':
    top, bot, perf = recommend()
    print("=== 申万行业轮动（20日夏普主信号）===\n")
    if top:
        print(f"🔥 强势: {', '.join([f'{n}(夏普{v:+.2f})' for n,v in top])}")
    if bot:
        print(f"⚠️ 弱势: {', '.join([f'{n}(夏普{v:+.2f})' for n,v in bot])}")
    if not perf.empty:
        print(f"\n全行业排名:\n{perf.head(15).to_string(index=False)}")
