#!/usr/bin/env python3
# /tmp/sav3/data_fetcher.py
# 数据获取层 — tushare（替代 baostock）
import os
import re
import time
import random
import hashlib
import logging
import pandas as pd
import tushare as ts
from config import CACHE_DIR, TUSHARE_TOKEN, INDEX_CODE

_pro = None
logger = logging.getLogger(__name__)

# ── tushare 限流防护（P0-3 修复）──
# 该 token 档位实测限流约 200 次/分钟，无 sleep 连续调用中段全失败且被静默吞掉。
# 所有 pro 调用统一经 _call_with_retry：节流 ≥0.35s/次 + 指数退避重试。
_API_MIN_INTERVAL = 0.35
_last_call_ts = [0.0]


def _throttle():
    wait = _API_MIN_INTERVAL - (time.time() - _last_call_ts[0])
    if wait > 0:
        time.sleep(wait)
    _last_call_ts[0] = time.time()


def _call_with_retry(fn, desc='', retries=4, base_delay=0.6):
    """节流 + 指数退避重试。全部失败返回 None（不抛异常，由调用方决定门禁）。"""
    for attempt in range(retries):
        _throttle()
        try:
            return fn()
        except Exception as e:
            delay = base_delay * (2 ** attempt)
            if attempt < retries - 1:
                logger.warning(f"API调用失败({desc}) 第{attempt + 1}/{retries}次: {e}; {delay:.1f}s后重试")
                time.sleep(delay)
            else:
                logger.error(f"API调用失败({desc}) 已达{retries}次上限: {e}")
    return None

# tushare 内部指数代码格式映射（config 用 'sh.000300'，tushare pro 用 '000300.SH'）
_INDEX_CODE_MAP = {
    'sh.000300': '000300.SH',
    'sh.000001': '000001.SH',
    'sh.000688': '000688.SH',
    'sh.000852': '000852.SH',
    'sz.399001': '399001.SZ',
    'sz.399006': '399006.SZ',
}


def _get_pro():
    global _pro
    if _pro is None:
        _pro = ts.pro_api(TUSHARE_TOKEN)
    return _pro


def _to_ts_index_code(index_code: str) -> str:
    """将配置风格或 tushare 风格指数代码统一转换为 tushare pro 风格。"""
    # 已经是 tushare 风格
    if index_code.endswith('.SH') or index_code.endswith('.SZ'):
        return index_code
    idx_ts = _INDEX_CODE_MAP.get(index_code)
    if idx_ts is not None:
        return idx_ts
    # fallback: 尝试旧逻辑
    idx_ts = index_code.replace('sh.', '') + '.SH'
    if '399' in index_code or 'sz.' in index_code:
        idx_ts = index_code.replace('sz.', '') + '.SZ'
    return idx_ts


def get_stock_basic(refresh: bool = False) -> pd.DataFrame:
    """获取全A股基础信息，带缓存"""
    cf = os.path.join(CACHE_DIR, 'ts_stock_basic.pkl')
    if not refresh and os.path.exists(cf):
        cache_age = pd.Timestamp.now() - pd.Timestamp.fromtimestamp(os.path.getmtime(cf))
        if cache_age.days < 7:
            return pd.read_pickle(cf)

    pro = _get_pro()
    # 分交易所拉取，合并
    dfs = []
    for exchange in ['SSE', 'SZSE', 'BSE']:
        df = _call_with_retry(lambda ex=exchange: pro.stock_basic(
            exchange=ex, list_status='L',
            fields='ts_code,symbol,name,area,industry,list_date'), desc=f'stock_basic {exchange}')
        if df is not None:
            dfs.append(df)
    if not dfs:
        raise ValueError("stock_basic 拉取失败")
    result = pd.concat(dfs, ignore_index=True)
    result['code'] = result['ts_code']  # 兼容老代码
    result.to_pickle(cf)
    return result


def get_all_codes(limit: int = None) -> list:
    """获取股票代码列表"""
    basic = get_stock_basic()
    codes = basic['ts_code'].unique().tolist()
    if limit and len(codes) > limit:
        random.seed(42)  # 固定种子保证可复现
        random.shuffle(codes)
        codes = codes[:limit]
    return codes


def _cleanup_stale_cache(prefix: str, keep: str = None, max_age_days: int = 14):
    """清理同类旧缓存文件（旧命名残留 + 超过 max_age_days 未更新的文件）。

    只在成功写入新缓存后调用，保证随时有可用的最新缓存。
    """
    now = time.time()
    keep_abs = os.path.abspath(keep) if keep else None
    for f in os.listdir(CACHE_DIR):
        if not f.startswith(prefix) or not f.endswith('.pkl'):
            continue
        path = os.path.join(CACHE_DIR, f)
        if keep_abs and os.path.abspath(path) == keep_abs:
            continue
        try:
            age_days = (now - os.path.getmtime(path)) / 86400
            if age_days > max_age_days:
                os.remove(path)
                logger.info(f"清理旧缓存: {f} ({age_days:.0f}天未更新)")
        except OSError:
            pass


def _seed_legacy_cache(pattern: re.Pattern):
    """从新命名缓存缺失时，迁移最新的旧命名缓存（含 end_date 的历史文件）作为增量种子。"""
    cands = [f for f in os.listdir(CACHE_DIR) if pattern.match(f)]
    if not cands:
        return None
    newest = max(cands, key=lambda f: os.path.getmtime(os.path.join(CACHE_DIR, f)))
    try:
        data = pd.read_pickle(os.path.join(CACHE_DIR, newest))
        logger.info(f"迁移旧命名缓存作为增量种子: {newest}")
        return data
    except Exception:
        return None


def _latest_trade_date(end_date) -> pd.Timestamp:
    """end_date 及之前最近一个交易日（用于缓存新鲜度校验）。"""
    try:
        dates = get_trade_dates(
            (pd.to_datetime(end_date) - pd.Timedelta(days=14)).strftime('%Y-%m-%d'),
            pd.to_datetime(end_date).strftime('%Y-%m-%d'))
        if dates:
            return pd.Timestamp(max(dates))  # trade_cal 返回降序，不能取 [-1]
    except Exception as e:
        logger.warning(f"trade_cal 获取失败，新鲜度校验降级: {e}")
    # 降级：容忍周末/小长假
    return pd.to_datetime(end_date) - pd.Timedelta(days=4)


def fetch_stock_daily(codes: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    拉取个股日K（tushare daily 接口，批量高效）
    返回 MultiIndex(date, code) DataFrame

    P0-3 修复：
    - 缓存文件名只含股票池标识（不含日期区间），跨日可命中；
    - 增量拉取缺失交易日（冷启动后每日仅需 1~2 次 API 调用）；
    - 全链路节流 + 重试；失败交易日占比 >2% 时不写缓存（防残缺缓存）。
    """
    n_codes = len(codes)
    # 缓存 key 加入股票代码哈希，避免不同代码组合但数量相同时缓存冲突
    code_hash = hashlib.sha256(','.join(sorted(codes)).encode()).hexdigest()[:12]
    cf = os.path.join(CACHE_DIR, f'ts_daily_{n_codes}_{code_hash}.pkl')

    cached = None
    if os.path.exists(cf):
        try:
            cached = pd.read_pickle(cf)
        except Exception:
            cached = None
    if cached is None:
        # 迁移旧命名缓存（ts_daily_{start}_{end}_{n}_{hash}.pkl）作为种子
        cached = _seed_legacy_cache(re.compile(
            rf'ts_daily_\d{{4}}-?\d{{2}}-?\d{{2}}_\d{{4}}-?\d{{2}}-?\d{{2}}_{n_codes}_{code_hash}\.pkl$'))

    start_ts = pd.Timestamp(start_date)
    latest_td = _latest_trade_date(end_date)

    # 命中：覆盖请求起点且尾部新鲜
    if cached is not None and not cached.empty:
        cached_dates = cached.index.get_level_values('date')
        if cached_dates.min() <= start_ts and cached_dates.max() >= latest_td:
            return cached

    pro = _get_pro()

    # 计算缺失的交易日
    want_dates = get_trade_dates(start_date, end_date)
    have_dates = set()
    if cached is not None and not cached.empty:
        have_dates = set(cached.index.get_level_values('date').strftime('%Y%m%d'))
    missing = [d for d in want_dates if d not in have_dates]

    all_data = []
    fetched = 0
    failed = 0
    if missing:
        logger.info(f"个股日K增量拉取: {len(missing)} 个交易日缺失")
        for dt_str in missing:
            df = _call_with_retry(lambda d=dt_str: pro.daily(trade_date=d),
                                  desc=f'daily {dt_str}')
            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(dt_str)
                all_data.append(df)
                fetched += 1
            else:
                failed += 1
        logger.info(f"  拉取完成: {fetched}/{len(missing)} 个交易日 (失败 {failed})")

    frames = []
    if cached is not None and not cached.empty:
        # 缓存帧列名归一化回 API 风格（ts_code/vol），避免与新拉取帧 concat 出重复列；
        # name 列丢弃，由下方 merge 统一重建（否则新行 name 为 NaN）
        c = cached.reset_index().rename(columns={'code': 'ts_code', 'volume': 'vol'})
        c = c.drop(columns=['name'], errors='ignore')
        frames.append(c)
    if all_data:
        frames.extend(all_data)
    if not frames:
        raise ValueError("No stock data fetched")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={'vol': 'volume'})

    # 合并股票名称（供发酵检测等下游使用）
    try:
        basic = get_stock_basic()
        if 'name' in basic.columns and 'name' not in combined.columns:
            combined = combined.merge(basic[['ts_code', 'name']], on='ts_code', how='left')
    except Exception:
        pass  # 降级：无名称也能运行，只是子方向分类可能不完整

    # 过滤到 codes 范围
    combined = combined[combined['ts_code'].isin(codes)]

    combined = combined.set_index(['date', 'ts_code']).sort_index()
    combined.index = combined.index.set_names(['date', 'code'])
    combined = combined[~combined.index.duplicated(keep='last')]

    # 补充 preclose 和 pct_chg
    if 'pre_close' not in combined.columns:
        combined['pre_close'] = combined.groupby('code')['close'].shift(1)
    if 'pct_chg' not in combined.columns:
        combined['pct_chg'] = combined.groupby('code')['close'].pct_change() * 100

    # 数值转换
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount', 'pct_chg']:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')

    # 完整性门禁：失败交易日占比 >2% 时不写缓存（沿用旧缓存，宁可缺不可错）
    if missing and failed / len(missing) > 0.02:
        logger.error(f"个股日K失败率 {failed}/{len(missing)} ({failed / len(missing):.1%}) > 2%，"
                     f"不写缓存（防残缺缓存，P0-3 门禁）")
    else:
        combined.to_pickle(cf)
        _cleanup_stale_cache('ts_daily_', keep=cf)
    return combined


def fetch_index_daily(start_date: str, end_date: str,
                      index_code: str = None, fields: list = None) -> pd.DataFrame:
    """
    拉取单指数日K。

    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        index_code: 指数代码（config 风格，如 'sh.000300'）。默认使用 config.INDEX_CODE。
        fields: 需要保留的列列表，默认 ['close']。

    Returns:
        DataFrame 以 date 为索引，包含指定 fields 列。
    """
    if index_code is None:
        index_code = INDEX_CODE
    if fields is None:
        fields = ['close']

    result = fetch_indices_daily(start_date, end_date, {index_code: index_code}, fields=fields)
    df = result.get(index_code)
    if df is None:
        raise ValueError(f"No index data for {index_code}")
    return df


def _fetch_index_range(ts_code: str, start_str: str, end_str: str, fields: list):
    """按月分段拉取单指数 [start_str, end_str]，全链路节流+重试。

    返回值三态：
    - None: 所有月份调用均失败（API 故障）
    - 空 DataFrame: 调用成功但该区间无数据（如指数尚未成立）
    - 非空 DataFrame: 正常数据
    """
    pro = _get_pro()
    all_data = []
    month_starts = pd.date_range(start=start_str, end=end_str, freq='MS')
    if len(month_starts) == 0 or month_starts[0] > pd.Timestamp(start_str):
        month_starts = month_starts.insert(0, pd.Timestamp(start_str))

    failed_months = 0
    for i, ms in enumerate(month_starts):
        me = min(month_starts[i + 1] - pd.Timedelta(days=1), pd.Timestamp(end_str)) \
            if i + 1 < len(month_starts) else pd.Timestamp(end_str)
        df = _call_with_retry(
            lambda s=ms.strftime('%Y%m%d'), e=me.strftime('%Y%m%d'):
                pro.index_daily(ts_code=ts_code, start_date=s, end_date=e),
            desc=f'index_daily {ts_code} {ms:%Y-%m}')
        if df is not None and not df.empty:
            df['date'] = pd.to_datetime(df['trade_date'])
            all_data.append(df)
        elif df is None:
            failed_months += 1

    # 全部月份都失败（异常）才算拉取失败；空响应是合法"无数据"
    if failed_months == len(month_starts) and len(month_starts) > 0:
        return None
    if not all_data:
        return pd.DataFrame()
    if failed_months:
        logger.warning(f"{ts_code}: {failed_months}/{len(month_starts)} 个月分段拉取失败")

    df = pd.concat(all_data, ignore_index=True)
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'vol', 'amount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.set_index('date').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    keep = [c for c in fields if c in df.columns]
    return df[keep if keep else ['close']]


def fetch_indices_daily(start_date: str, end_date: str,
                        index_codes: dict, fields: list = None,
                        refresh: bool = False) -> dict:
    """
    拉取多指数日K。

    P0-3 修复：
    - 缓存文件名只含指数集合哈希（不含日期区间），跨日可命中；
    - 增量拉取：缓存基础上只补头/尾缺失区间（日常运行每指数 1~2 次调用）；
    - 全链路节流 + 指数退避重试（tushare 限流约 200 次/分钟）；
    - 完整性门禁：任一指数缺失或尾部不新鲜则不写缓存（残缺数据绝不落盘），
      但仍返回已得数据，由日报健康区亮红灯。

    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        index_codes: dict {输出名称: 指数代码（config 风格或 tushare 风格）}
        fields: 需要保留的列列表，默认 ['close']；传 None 或 ['close'] 时行为与旧版一致。
        refresh: 是否强制刷新缓存。

    Returns:
        dict {输出名称: DataFrame(date 为索引)}
    """
    if fields is None:
        fields = ['close']
    start_str = pd.to_datetime(start_date).strftime('%Y%m%d')
    end_str = pd.to_datetime(end_date).strftime('%Y%m%d')
    code_hash = hashlib.sha256(','.join(sorted(index_codes.values())).encode()).hexdigest()[:12]
    # 缓存文件名不含日期区间（旧命名含 end_date → 跨日零命中 → 每天全量重拉再截断）
    cf = os.path.join(CACHE_DIR, f'ts_idx_{code_hash}.pkl')
    meta_cf = os.path.join(CACHE_DIR, f'ts_idx_{code_hash}.meta.json')

    # 头部穷尽标记：指数成立日晚于请求起点时，空拉取一次后永久标记，避免每日无效重拉
    head_exhausted = set()
    if os.path.exists(meta_cf):
        try:
            import json as _json
            with open(meta_cf) as _f:
                head_exhausted = set(_json.load(_f).get('head_exhausted', []))
        except Exception:
            pass

    cached = None
    if not refresh:
        if os.path.exists(cf):
            try:
                cached = pd.read_pickle(cf)
            except Exception:
                cached = None
        if cached is None:
            # 迁移旧命名缓存（ts_idx_{start}_{end}_{hash}.pkl）作为增量种子
            cached = _seed_legacy_cache(re.compile(
                rf'ts_idx_\d{{8}}_\d{{8}}_{code_hash}\.pkl$'))

    start_ts = pd.Timestamp(start_str)
    head_tolerance = start_ts + pd.Timedelta(days=12)  # 容忍长假（春节等）
    latest_td = _latest_trade_date(end_date)

    def _head_ok(name, df):
        return df.index.min() <= head_tolerance or name in head_exhausted

    # 命中：每个指数都覆盖请求起点且尾部新鲜
    if isinstance(cached, dict):
        fresh = all(
            isinstance(cached.get(name), pd.DataFrame) and not cached[name].empty
            and _head_ok(name, cached[name])
            and cached[name].index.max() >= latest_td
            for name in index_codes)
        if fresh:
            return cached

    result = {}
    incomplete = []
    for name, code in index_codes.items():
        ts_code = _to_ts_index_code(code)
        existing = cached.get(name) if isinstance(cached, dict) else None
        if existing is not None and (existing.empty or not isinstance(existing, pd.DataFrame)):
            existing = None

        # 计算头/尾缺失区间 (start, end, is_head)
        ranges = []
        if existing is None:
            ranges.append((start_str, end_str, True))
        else:
            if existing.index.min() > head_tolerance and name not in head_exhausted:
                ranges.append((start_str,
                               (existing.index.min() - pd.Timedelta(days=1)).strftime('%Y%m%d'),
                               True))
            if existing.index.max() < latest_td:
                ranges.append(((existing.index.max() + pd.Timedelta(days=1)).strftime('%Y%m%d'),
                               end_str, False))

        pieces = [existing] if existing is not None else []
        fetch_failed = False
        for rs, re_, is_head in ranges:
            new = _fetch_index_range(ts_code, rs, re_, fields)
            if new is None:
                fetch_failed = True
            elif new.empty:
                # 空响应=该区间无数据。头部区间 → 标记穷尽（指数成立日晚于请求起点），不再重拉
                if is_head:
                    head_exhausted.add(name)
            else:
                pieces.append(new)

        if not pieces:
            logger.error(f"指数 {name} ({ts_code}) 无数据（拉取失败且无缓存）")
            incomplete.append(name)
            continue

        merged = pd.concat(pieces)
        merged = merged[~merged.index.duplicated(keep='last')].sort_index()
        result[name] = merged

        if fetch_failed or merged.index.max() < latest_td:
            incomplete.append(name)
            logger.error(f"指数 {name} 数据不完整: {merged.index.min().date()} ~ "
                         f"{merged.index.max().date()} (要求尾部 ≥{latest_td.date()})")

    if not result:
        raise ValueError("No index data")

    # 完整性门禁：任一指数缺失/尾部不新鲜 → 不写缓存（防残缺缓存复发）
    if incomplete:
        logger.error(f"指数数据不完整: {incomplete} —— 不写缓存（P0-3 门禁）")
    else:
        pd.to_pickle(result, cf)
        try:
            import json as _json
            with open(meta_cf, 'w') as _f:
                _json.dump({'head_exhausted': sorted(head_exhausted)}, _f)
        except Exception:
            pass
        _cleanup_stale_cache('ts_idx_', keep=cf)
    return result


def fetch_close_by_code(ts_code: str, target_date, lookback_days: int = 60) -> float | None:
    """
    拉取单只股票在 target_date 附近的最近有效收盘价。

    Args:
        ts_code: tushare 风格代码，如 '000001.SZ'
        target_date: 目标日期（str 或 Timestamp）
        lookback_days: 向前回看天数

    Returns:
        最近有效收盘价；无数据返回 None。
    """
    pro = _get_pro()
    target_dt = pd.to_datetime(target_date)
    start_dt = target_dt - pd.Timedelta(days=lookback_days)
    start_str = start_dt.strftime('%Y%m%d')
    end_str = target_dt.strftime('%Y%m%d')
    df = _call_with_retry(
        lambda: pro.daily(ts_code=ts_code, start_date=start_str, end_date=end_str),
        desc=f'daily {ts_code}')
    if df is None or df.empty:
        return None
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    valid = df[df['close'] > 0]
    if valid.empty:
        return None
    return float(valid['close'].iloc[-1])


def fetch_holdings_closes(ts_codes: list, target_date, lookback_days: int = 60) -> dict:
    """
    拉取持仓股（含 ST/停牌等不在选股池的）近期日K，返回 {ts_code: DataFrame}。

    每只股票单独小缓存（ts_hold_{code}.pkl，文件名不含日期），增量补拉：
    - 缓存尾部距今 ≤4 天（覆盖周末/小长假）→ 直接命中，零 API 调用；
    - 否则只补拉 (缓存尾日+1) ~ target_date 的缺口并合并落盘；
    - API 失败时回退旧缓存（宁可旧不可无）；
    - 返回数据始终过滤到 trade_date ≤ target_date（防回测前视）。

    ST 股被 data_preparation 过滤、停牌股不在选股池，原先每天全量 pro.daily
    补拉 60 天；改为小缓存后日常运行对这些股票零/一次 API 调用。
    """
    pro = _get_pro()
    target_dt = pd.to_datetime(target_date)
    end_str = target_dt.strftime('%Y%m%d')
    result = {}

    for ts_code in ts_codes:
        cf = os.path.join(CACHE_DIR, f'ts_hold_{ts_code.replace(".", "_")}.pkl')
        cached = None
        if os.path.exists(cf):
            try:
                cached = pd.read_pickle(cf)
            except Exception:
                cached = None

        df_new = None
        if cached is not None and not cached.empty:
            cached = cached.copy()
            cached['trade_date'] = cached['trade_date'].astype(str)
            last_ts = pd.Timestamp(cached['trade_date'].max())
            if (target_dt - last_ts).days <= 4:
                df_new = cached  # 尾部新鲜，命中
            else:
                fetch_start = (last_ts + pd.Timedelta(days=1)).strftime('%Y%m%d')
                incr = _call_with_retry(
                    lambda s=fetch_start: pro.daily(ts_code=ts_code,
                                                    start_date=s, end_date=end_str),
                    desc=f'hold {ts_code}')
                if incr is not None and not incr.empty:
                    df_new = pd.concat([cached, incr], ignore_index=True)
                elif incr is None:
                    df_new = cached  # 拉取失败回退旧缓存
                else:
                    df_new = cached  # 缺口区间无数据（停牌/未复牌），沿用缓存
        else:
            start_str = (target_dt - pd.Timedelta(days=lookback_days)).strftime('%Y%m%d')
            df_new = _call_with_retry(
                lambda s=start_str: pro.daily(ts_code=ts_code,
                                              start_date=s, end_date=end_str),
                desc=f'hold {ts_code}')

        if df_new is None or df_new.empty:
            continue

        df_new['trade_date'] = df_new['trade_date'].astype(str)
        df_new['close'] = pd.to_numeric(df_new['close'], errors='coerce')
        df_new = (df_new.drop_duplicates(subset='trade_date', keep='last')
                        .sort_values('trade_date')
                        .tail(120)
                        .reset_index(drop=True))
        try:
            df_new.to_pickle(cf)
        except Exception as e:
            logger.warning(f"持仓小缓存写入失败 {ts_code}: {e}")

        # 防前视：只返回 target_date 及之前的记录
        out = df_new[df_new['trade_date'] <= end_str]
        if not out.empty:
            result[ts_code] = out
    return result


def fetch_moneyflow_hsgt(start_date: str, end_date: str) -> pd.DataFrame:
    """拉取北向资金"""
    pro = _get_pro()
    df = _call_with_retry(
        lambda: pro.moneyflow_hsgt(start_date=start_date.replace('-', ''),
                                   end_date=end_date.replace('-', '')),
        desc='moneyflow_hsgt')
    if df is not None and not df.empty:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date').sort_index()
    return df


def get_trade_dates(start_date: str, end_date: str) -> list:
    """获取交易日列表"""
    pro = _get_pro()
    df = _call_with_retry(
        lambda: pro.trade_cal(exchange='SSE', start_date=start_date.replace('-', ''),
                              end_date=end_date.replace('-', '')),
        desc='trade_cal')
    if df is None or df.empty:
        raise ValueError(f"trade_cal 拉取失败: {start_date}~{end_date}")
    dates = df[df['is_open'] == 1]['cal_date'].tolist()
    return dates
