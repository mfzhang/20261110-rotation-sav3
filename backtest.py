# ARC 通用回测引擎 — 消除4处重复，通过 score_modifier 支持自定义因子增强
import os
import logging
import pandas as pd
import numpy as np
from scipy import stats
from config import *
from crisis_detector import CrisisDetector
from score_ranker import compute_regime_weights

import factor_pipeline
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# 退市股月中收益保守计提（A股退市整理期平均跌幅 50%+ 的量级；
# 不计提 = 把退市损失从回测中静默抹掉，即幸存者偏差）
DELIST_HAIRCUT = -0.5


def prepare_backtest_inputs(start, end, limit, skip_financials=True):
    """统一数据准备管线：拉取数据、计算因子、获取指数数据。"""
    s_data = (pd.to_datetime(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    from data_preparation import prepare_data
    from factor_calculator import calculate_factors
    from data_fetcher import fetch_index_daily
    daily, idx_ret, industry_map, _ = prepare_data(
        s_data, end, stock_limit=limit, skip_financials=skip_financials
    )
    factors = calculate_factors(daily, idx_ret)
    index_df = fetch_index_daily(s_data, end)
    return daily, idx_ret, industry_map, index_df, factors


# ══════════════════════════════════════════════════════════════
# 含退市股（list_status='D'）保守口径 — 2026-07-19 模型层修复
#
# 标准口径的幸存者偏差链条：stock_basic 只拉 list_status='L' →
# 股票池不含任何已退市股；且回测循环要求"下月调仓日仍有价格"才允许
# 持有 → 月中退市股被静默剔除，退市损失从未进入业绩。
# 保守口径：并入窗口内曾上市的退市股（ST 不过滤——戴帽是退市前奏），
# 持有期间退市的按 DELIST_HAIRCUT 计提当月收益。
# ══════════════════════════════════════════════════════════════

def _fetch_delisted_basic(refresh: bool = False) -> pd.DataFrame:
    """list_status='D' 退市股基础信息（7 天磁盘缓存，对齐 get_stock_basic）。"""
    cf = os.path.join(CACHE_DIR, 'ts_stock_basic_D.pkl')
    if not refresh and os.path.exists(cf):
        age = pd.Timestamp.now() - pd.Timestamp.fromtimestamp(os.path.getmtime(cf))
        if age.days < 7:
            return pd.read_pickle(cf)
    from data_fetcher import _get_pro, _call_with_retry
    pro = _get_pro()
    dfs = []
    for exchange in ['SSE', 'SZSE', 'BSE']:
        df = _call_with_retry(lambda ex=exchange: pro.stock_basic(
            exchange=ex, list_status='D',
            fields='ts_code,symbol,name,area,industry,list_date,delist_date'),
            desc=f'stock_basic D {exchange}')
        if df is not None:
            dfs.append(df)
    if not dfs:
        raise ValueError("stock_basic(list_status='D') 拉取失败")
    result = pd.concat(dfs, ignore_index=True)
    result.to_pickle(cf)
    return result


def _fetch_delisted_daily(codes: list, start_date: str, end_date: str) -> pd.DataFrame:
    """按股拉取退市股日K（pro.daily ts_code 接口），缓存 ts_daily_delisted.pkl。

    增量逻辑：缓存中缺失的代码逐只拉取（每只 1 次 API 调用，全链路节流重试）；
    已退市股票历史不再变化，缓存长期有效；新退市股（代码不在缓存中）自动补拉。
    返回 MultiIndex(date, code) DataFrame。
    """
    from data_fetcher import _get_pro, _call_with_retry
    cf = os.path.join(CACHE_DIR, 'ts_daily_delisted.pkl')
    cached = None
    if os.path.exists(cf):
        try:
            cached = pd.read_pickle(cf)
        except Exception:
            cached = None

    have = set()
    if cached is not None and not cached.empty:
        have = set(cached.index.get_level_values('code').unique())
    missing = [c for c in codes if c not in have]

    frames = []
    if cached is not None and not cached.empty:
        frames.append(cached.reset_index())

    if missing:
        pro = _get_pro()
        fetched = 0
        for code in missing:
            df = _call_with_retry(lambda c=code: pro.daily(
                ts_code=c,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')),
                desc=f'delisted daily {code}')
            if df is None or df.empty:
                continue
            df = df.rename(columns={'ts_code': 'code', 'vol': 'volume'})
            df['date'] = pd.to_datetime(df['trade_date'])
            frames.append(df)
            fetched += 1
        logger.info(f"退市股日K: 新拉取 {fetched}/{len(missing)} 只")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['date', 'code'], keep='last')
    combined = combined.set_index(['date', 'code']).sort_index()
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount', 'pct_chg']:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')

    if missing:  # 有新拉取才写缓存（避免无意义覆写）
        combined.to_pickle(cf)
    return combined


def _clean_delisted_daily(daily_d: pd.DataFrame, d_basic: pd.DataFrame,
                          index_ret: pd.Series) -> pd.DataFrame:
    """退市股日K清洗，对齐 prepare_data 输出结构。

    与 prepare_data 的规则差异（均为保守取向，防再次引入幸存者偏差）：
    - 不过滤 ST：戴帽是退市前奏，剔除 ST 等于剔除大部分退市损失
    - 上市时长过滤保留（MIN_LIST_DAYS）
    - 流动性只做绝对成交额门槛（LIQUIDITY_MIN_AMOUNT），不做相对分位——
      退市股临终成交萎缩，相对分位会把它们系统性剔除
    - 涨跌停 tradable 标记与 prepare_data 同函数同口径
    """
    from data_preparation import _price_limit_threshold

    daily = daily_d.reset_index()
    basic = d_basic.set_index('ts_code')
    daily['name'] = daily['code'].map(basic['name'])
    daily['list_date'] = pd.to_datetime(daily['code'].map(basic['list_date']),
                                        errors='coerce')
    daily['list_days'] = (daily['date'] - daily['list_date']).dt.days
    daily = daily[daily['list_days'] >= MIN_LIST_DAYS]
    daily = daily.sort_values(['code', 'date'])

    # 涨跌停标记（同 prepare_data 口径）
    if STOP_FILTER:
        thr = _price_limit_threshold(daily['code'], daily['name'].fillna(''),
                                     daily['date'])
        daily['tradable'] = (daily['pct_chg'] < thr) & (daily['pct_chg'] > -thr)
    else:
        daily['tradable'] = True

    # 流动性：仅绝对成交额门槛（理由见 docstring）
    if 'amount' in daily.columns:
        daily['avg_amount'] = daily.groupby('code')['amount'].transform(
            lambda x: x.rolling(LIQUIDITY_WINDOW, min_periods=1).mean())
        daily = daily[daily['avg_amount'] >= LIQUIDITY_MIN_AMOUNT]

    daily = daily.set_index(['date', 'code']).sort_index()

    # 衍生指标（与 prepare_data 同定义，保证因子口径一致）
    daily['ret'] = daily.groupby('code')['close'].pct_change()
    daily['high_52w'] = daily.groupby('code')['high'].transform(
        lambda x: x.rolling(HIGH_LOW_WINDOW, min_periods=1).max())
    daily['low_52w'] = daily.groupby('code')['low'].transform(
        lambda x: x.rolling(HIGH_LOW_WINDOW, min_periods=1).min())
    daily['ret_20d'] = daily.groupby('code')['close'].transform(
        lambda x: x.pct_change(periods=REL_STRENGTH_DAYS))
    daily['index_ret'] = daily.index.get_level_values('date').map(index_ret)
    daily['np_margin'] = np.nan
    daily['cash_to_np'] = np.nan
    daily['leverage'] = np.nan
    return daily


def prepare_backtest_inputs_conservative(start, end, limit, skip_financials=True,
                                         delist_haircut=DELIST_HAIRCUT):
    """含退市股（list_status='D'）保守口径的数据准备。

    在标准 L 池基础上，并入回测窗口内曾上市、后已退市的股票：
    - 因子在合并宇宙上统一计算（与标准口径完全同函数同参数）
    - 行业映射并入退市股（否则中性化把退市股归为 NaN→rank 0.5 中性，
      等于再次把它们从选股中静默剔除）
    - 返回 (daily, idx_ret, industry_map, index_df, factors, delist_info)，
      其中 delist_info 供 run_backtest_loop/backtest_dual_mode 做退市计提
    """
    s_data = (pd.to_datetime(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    from data_preparation import prepare_data
    from factor_calculator import calculate_factors
    from data_fetcher import fetch_index_daily

    # 1) 标准 L 池（与 prepare_backtest_inputs 同口径、同 seed 股票池）
    daily_l, idx_ret, industry_map, index_df = prepare_data(
        s_data, end, stock_limit=limit, skip_financials=skip_financials)

    # 2) 窗口内曾上市的退市股
    d_basic = _fetch_delisted_basic()
    d_basic = d_basic.copy()
    d_basic['delist_date'] = pd.to_datetime(d_basic['delist_date'], errors='coerce')
    d_basic['list_date'] = pd.to_datetime(d_basic['list_date'], errors='coerce')
    in_window = d_basic[
        (d_basic['delist_date'] >= pd.Timestamp(s_data)) &
        (d_basic['list_date'] <= pd.Timestamp(end))]
    l_codes = set(daily_l.index.get_level_values('code'))
    codes_d = [c for c in in_window['ts_code'].tolist() if c not in l_codes]
    logger.info(f"保守口径: {len(codes_d)} 只窗口内退市股并入股票池")

    delist_dates = {}
    daily = daily_l
    if codes_d:
        daily_d_raw = _fetch_delisted_daily(codes_d, s_data, end)
        if not daily_d_raw.empty:
            daily_d = _clean_delisted_daily(daily_d_raw, in_window, idx_ret)
            daily = pd.concat([daily_l, daily_d])
            daily = daily[~daily.index.duplicated(keep='first')].sort_index()
            # 行业映射并入退市股（防中性化丢弃）
            industry_map = pd.concat([
                industry_map,
                in_window.set_index('ts_code')['industry'].dropna()
            ])
            industry_map = industry_map[~industry_map.index.duplicated(keep='first')]
        else:
            logger.warning("退市股日K为空，退回标准口径")
        delist_dates = {r['ts_code']: r['delist_date']
                        for _, r in in_window.iterrows()
                        if pd.notna(r['delist_date'])}

    # 3) 合并宇宙统一计算因子（与标准口径同函数）
    factors = calculate_factors(daily, idx_ret)
    index_df = fetch_index_daily(s_data, end)

    delist_info = {
        'delist_dates': delist_dates,
        'haircut': delist_haircut,
        'n_delisted': len(codes_d),
    }
    return daily, idx_ret, industry_map, index_df, factors, delist_info


def compute_month_ends(dates):
    """从日期序列中提取每月最后一个交易日（已排序）。"""
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    return sorted(pd.Series(dates).groupby(dates.to_period('M')).last().values)


def compute_factor_ic_series(factor_values, daily_data, month_ends,
                             min_stocks=10, method='spearman'):
    """计算单因子的月度 Rank IC 序列。

    参数
    ----
    factor_values : pd.Series, MultiIndex(date, code)
        因子值。
    daily_data : pd.DataFrame, MultiIndex(date, code)
        需包含 'close' 列。
    month_ends : list of Timestamp
        月末调仓日期列表，函数将用相邻两个月末计算 forward 1-month 收益。
    min_stocks : int
        每期最少共同股票数，低于则跳过。
    method : str
        目前仅支持 'spearman'。

    返回
    ----
    list of float : 每期 IC 值（跳过的不包含）。
    """
    ic_list = []
    for i in range(len(month_ends) - 1):
        dt, nxt = month_ends[i], month_ends[i + 1]
        try:
            f_slice = factor_values.xs(dt, level='date')
            cs = daily_data.xs(dt, level='date')['close']
            ce = daily_data.xs(nxt, level='date')['close']
        except Exception:
            continue
        fwd_ret = ce / cs - 1.0
        f_clean = f_slice.dropna()
        r_clean = fwd_ret.dropna()
        common = f_clean.index.intersection(r_clean.index)
        if len(common) < min_stocks:
            continue
        try:
            if method == 'spearman':
                ic, _ = stats.spearmanr(f_clean.loc[common], r_clean.loc[common])
            else:
                ic, _ = stats.pearsonr(f_clean.loc[common], r_clean.loc[common])
            ic_list.append(ic)
        except Exception:
            continue
    return ic_list


def compute_performance_metrics(returns_df: pd.DataFrame) -> dict:
    """从月度收益序列计算标准绩效指标（含 bench_ret 列时输出基准对照）。"""
    if returns_df is None or len(returns_df) == 0:
        return None
    df = returns_df.copy()
    if 'date' in df.columns:
        df = df.set_index('date')
    total = (1 + df['ret']).prod() - 1
    yrs = max((df.index[-1] - df.index[0]).days / 365, 0.01)
    ann = (1 + total) ** (1 / yrs) - 1
    sh = df['ret'].mean() / (df['ret'].std() + 1e-10) * np.sqrt(12)
    cum = (1 + df['ret']).cumprod()
    mdd = (cum / cum.expanding().max() - 1).min()
    crisis_count = df['mode'].value_counts().get('crisis', 0) if 'mode' in df.columns else 0
    result = {
        'annual_return': ann, 'sharpe': sh, 'max_drawdown': mdd,
        'total_return': total, 'n_periods': len(df), 'crisis_periods': crisis_count,
        'returns_df': df,
    }
    # 基准对照（沪深300 同窗口）
    if 'bench_ret' in df.columns and df['bench_ret'].notna().any():
        b = df['bench_ret'].dropna()
        b_total = (1 + b).prod() - 1
        b_yrs = max((b.index[-1] - b.index[0]).days / 365, 0.01)
        result['bench_annual_return'] = (1 + b_total) ** (1 / b_yrs) - 1
        result['bench_total_return'] = b_total
        result['excess_annual'] = ann - result['bench_annual_return']
    return result


def _prepare_backtest_data(factors_df, daily_data, index_df, start_date, end_date,
                            industry_map=None):
    """预处理因子 + 构建月度调仓日期列表。返回 (factors_used, month_ends)。"""
    if industry_map is not None:
        factors_used = factor_pipeline.pipeline_process(factors_df, industry_map)
    else:
        factors_used = factors_df

    raw_dates = factors_used.index.get_level_values('date').unique()
    dates = pd.DatetimeIndex(sorted(raw_dates))
    month_ends = pd.Series(dates).groupby(dates.to_period('M')).last().values
    month_ends = sorted([d for d in month_ends
                         if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)])
    return factors_used, month_ends


def run_backtest_loop(factors_used, daily_data, index_df, month_ends,
                      normal_weights, crisis_weights, detector,
                      score_modifier=None, exposure_fn=None, top_n_fn=None,
                      top_n=TOP_N, with_cost=True,
                      daily_crisis_check=True, dynamic_weights=DYNAMIC_FACTOR_WEIGHTS,
                      apply_crisis_exposure=True,
                      delist_dates=None, delist_haircut=DELIST_HAIRCUT):
    """
    通用月度调仓回测循环（与生产同口径，P1-7 修复）。

    与生产的口径统一：
    - 评分：截面 rank 分位加权（原 z 分数加权，与生产不一致）
    - 危机检测：每日检测信号、月末执行调仓（原只在月末 check，
      exit_days=5 实际≈5个月，cooling_days 恒过，危机防御从未被真实测量）
    - 因子权重：compute_regime_weights 按市场状态动态配权（同生产）
    - 涨跌停：买入日 tradable=False 的股票不可买（同生产）
    - 危机期收益按 CRISIS_EXPOSURE 缩仓（与生产建议口径一致）
    - 成本：按实际换手率计费（原每月全额双边，高估 2~3 倍）
    - 退市（保守口径，delist_dates 提供时启用）：持有期间退市的股票
      按 delist_haircut 计提当月收益，不再被"下月无价格"静默剔除

    score_modifier(score, f_slice, dt, current_mode, extra_info) → modified_score
        返回 None 表示不修改（默认等权加权后直接排名）。
    exposure_fn(score, f_slice, dt, current_mode, extra_info) → modified_score
        仓位/风险暴露调整，例如根据 PSM 信号缩放得分。
    top_n_fn(dt, current_mode) → int
        动态选股数量，未提供则使用固定 top_n。
    delist_dates: dict {code: 退市日期} | None
        None=标准口径（现状）；提供时启用含退市股保守口径。
    delist_haircut: 退市当月收益计提（默认 DELIST_HAIRCUT=-0.5）

    返回 (returns_df, crisis_flags, extra_info)
    """
    factor_cols = [c for c in factors_used.columns]
    port_rets = []
    crisis_flags = []
    current_mode = 'normal'
    extra_info = {}  # 供 score_modifier/exposure_fn 写入额外统计
    prev_valid = None

    # ── 每日粒度危机检测：全历史逐日回放状态机，月末查表 ──
    idx_close_all = index_df['close'] if not index_df.empty else pd.Series(dtype=float)
    mode_series = None
    if daily_crisis_check and len(idx_close_all) > 0:
        modes = {}
        for d in idx_close_all.index:
            if d > month_ends[-1]:
                break
            m, _ = detector.check(idx_close_all.loc[:d], d)
            modes[d] = m
        mode_series = pd.Series(modes).sort_index()
        extra_info['crisis_days'] = int((mode_series == 'crisis').sum())

    for i in range(len(month_ends) - 1):
        dt, nxt = month_ends[i], month_ends[i + 1]

        # 危机模式：每日检测序列在 dt 当日的状态（月末执行）
        if mode_series is not None and len(mode_series) > 0:
            m = mode_series.asof(dt)
            if pd.notna(m):
                current_mode = m
        elif len(idx_close_all) > 252:
            # 兼容旧路径（daily_crisis_check=False）：仅月末 check
            idx_close = idx_close_all.loc[:dt]
            mode, emergency = detector.check(idx_close, dt)
            current_mode = mode

        try:
            f_slice = factors_used.xs(dt, level='date')
            cs = daily_data.xs(dt, level='date')['close']
            ce = daily_data.xs(nxt, level='date')['close']
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Skipping month {dt}: {e}")
            continue

        # 因子权重：与生产同函数按市场状态动态配权
        if dynamic_weights:
            w_dict, _regime = compute_regime_weights(idx_close_all.loc[:dt], current_mode)
            w = pd.Series(w_dict)
        else:
            w = crisis_weights if current_mode == 'crisis' else normal_weights

        # 买入日涨跌停不可交易过滤（P1-5，与生产同口径）
        blocked = None
        if 'tradable' in daily_data.columns:
            try:
                tr = daily_data.xs(dt, level='date')['tradable']
                blocked = set(tr[~tr].index)
            except KeyError:
                blocked = None

        # 计算 ARC 得分：截面 rank 分位加权（与生产 compute_composite_score 同口径）
        score = pd.Series(0.0, index=f_slice.index)
        for fac in factor_cols:
            if fac in f_slice.columns and fac in w.index and w[fac] != 0:
                rank_pct = f_slice[fac].rank(pct=True)
                score += w[fac] * rank_pct.fillna(0.5)

        # 可注入的得分修改器（如 C* 行业动量修正）
        if score_modifier is not None:
            score = score_modifier(score, f_slice, dt, current_mode, extra_info)

        # 可注入的仓位/暴露调整（如 PSM 联动降仓）
        if exposure_fn is not None:
            score = exposure_fn(score, f_slice, dt, current_mode, extra_info)

        # 动态选股数量
        this_top_n = top_n_fn(dt, current_mode) if top_n_fn is not None else top_n

        def _delisted_in_window(c):
            """该股是否在 (dt, nxt] 窗口内退市（保守口径下允许买入并计提损失）。"""
            if delist_dates is None:
                return False
            dd = delist_dates.get(c)
            return dd is not None and dt <= dd <= nxt

        def _buyable(codes):
            # 标准口径：买入日与下个调仓日都有价格才可买——"下月无价格"的退市股
            # 被静默剔除，这正是幸存者偏差的来源之一。
            # 保守口径（delist_dates 提供时）：月中退市股允许买入，
            # 收益在下方按 delist_haircut 计提。
            out = []
            for c in codes:
                if c not in cs.index:
                    continue
                if blocked is not None and c in blocked:
                    continue
                if c in ce.index or _delisted_in_window(c):
                    out.append(c)
            return out

        ranked = score.dropna().nlargest(this_top_n)
        valid = _buyable(ranked.index)
        if not valid:
            # 降级：score_modifier/exposure_fn 可能改变了排名，新 Top50 含下期缺失的股票
            # 从全量可交易股票中选 top_n，确保不出空期
            tradable = _buyable(score.index)
            if len(tradable) == 0:
                continue
            valid = score.loc[tradable].nlargest(this_top_n).index.tolist()

        # 交易成本：按实际换手率计费（原注释自述"未乘换手率"——月度换仓
        # 通常只换 30~50%，全额双边成本高估 2~3 倍）
        if with_cost:
            full_roundtrip = (BACKTEST_TRADING_COST * 2 + BACKTEST_STAMP_DUTY
                              + BACKTEST_SLIPPAGE * 2)
            if prev_valid:
                turnover = 1 - len(set(valid) & set(prev_valid)) / max(len(valid), 1)
            else:
                turnover = 1.0
            roundtrip_cost = full_roundtrip * turnover
        else:
            roundtrip_cost = 0.0
        prev_valid = list(valid)

        if delist_dates is None:
            # 标准口径（原逻辑）：要求买/卖日价格均存在且 >0
            safe = (cs[valid] > 0) & (ce[valid] > 0)
            ret = (ce[valid][safe] / cs[valid][safe] - 1).mean() - roundtrip_cost
        else:
            # 保守口径：月中退市股按 delist_haircut 计提当月收益
            rets = []
            for c in valid:
                p0 = cs.get(c)
                if p0 is None or not (p0 > 0):
                    continue
                p1 = ce.get(c)
                if p1 is not None and p1 > 0:
                    rets.append(p1 / p0 - 1)
                elif _delisted_in_window(c):
                    # 月中退市：下月无价格不再静默剔除，按 haircut 保守计提
                    # （不计提 = 把退市损失从回测中抹掉，即幸存者偏差）
                    rets.append(delist_haircut)
                    extra_info['delist_hits'] = extra_info.get('delist_hits', 0) + 1
            if not rets:
                continue
            ret = float(np.mean(rets)) - roundtrip_cost
        # 危机期仓位缩放（与生产"危机降至 40% 仓位"建议同口径）
        if apply_crisis_exposure and current_mode == 'crisis':
            ret = ret * CRISIS_EXPOSURE
        port_rets.append({'date': nxt, 'ret': ret, 'mode': current_mode})
        crisis_flags.append(current_mode)

    returns_df = pd.DataFrame(port_rets) if port_rets else None

    # 基准对照列：沪深300 同窗口月度收益（口径统一的业绩锚）
    if returns_df is not None and len(idx_close_all) > 0:
        bench = idx_close_all.reindex(pd.DatetimeIndex(month_ends)).ffill()
        bench_mret = bench.pct_change()
        returns_df['bench_ret'] = returns_df['date'].map(bench_mret)

    return returns_df, crisis_flags, extra_info


def backtest_dual_mode(factors_df, daily_data, index_df, start_date, end_date,
                       industry_map=None, roll_window=12, score_modifier=None,
                       delist_info=None):
    """双模式回测：固定权重（正常 0.5/0.5 + 危机 1.0/0.0）。

    ⚠️ 已知局限：行业分类为静态快照（来自最新 stock_basic），
    所有历史日期均使用最新的行业归属做中性化，存在前视偏差风险。

    delist_info: prepare_backtest_inputs_conservative 的输出（含退市股保守口径）；
                 None 时为标准口径（幸存者偏差未修正）。
    """
    delist_dates = (delist_info or {}).get('delist_dates')
    delist_haircut = (delist_info or {}).get('haircut', DELIST_HAIRCUT)
    if delist_info:
        print(f"⚠️  [保守口径] 含退市股 {delist_info.get('n_delisted', '?')} 只，"
              f"月中退市按 {delist_haircut:.0%} 计提（防幸存者偏差）")
    print("⚠️  [前视偏差] 行业分类使用最新快照，历史时点的实际行业归属可能不同。")
    factors_used, month_ends = _prepare_backtest_data(
        factors_df, daily_data, index_df, start_date, end_date, industry_map)

    factor_cols = [c for c in factors_used.columns]
    normal_weights = pd.Series({f: FACTOR_WEIGHTS.get(f, 0) for f in factor_cols})
    crisis_weights = pd.Series({f: CRISIS_WEIGHTS.get(f, 0) for f in factor_cols})
    print(f"\n>>> 回测 (动态配权={'开' if DYNAMIC_FACTOR_WEIGHTS else '关'})")
    print(f"  基准权重: {dict(normal_weights)}  危机: {dict(crisis_weights)}"
          f"{'  [正常期实际权重按市场状态动态调整]' if DYNAMIC_FACTOR_WEIGHTS else ''}")

    detector = CrisisDetector(index_drawdown=CRISIS_IDX_DRAWDOWN, vol_ratio=CRISIS_VOL_RATIO,
                              exit_drawdown=CRISIS_EXIT_DRAWDOWN,
                              exit_vol_ratio=CRISIS_EXIT_VOL_RATIO,
                              exit_days=CRISIS_EXIT_DAYS,
                              cooling_days=CRISIS_COOLING_DAYS,
                              stock_stoploss=CRISIS_STOCK_STOPLOSS,
                              fresh_start=True)

    returns_df, crisis_flags, extra_info = run_backtest_loop(
        factors_used, daily_data, index_df, month_ends,
        normal_weights, crisis_weights, detector,
        score_modifier=score_modifier,
        delist_dates=delist_dates, delist_haircut=delist_haircut)

    if returns_df is None or len(returns_df) == 0:
        return None

    result = compute_performance_metrics(returns_df)
    print(f"\n>>> 双模式回测结果 (与生产同口径: rank评分/每日危机检测/动态配权/换手率成本):")
    print(f"  年化收益: {result['annual_return']*100:.2f}%  "
          f"夏普: {result['sharpe']:.2f}  "
          f"最大回撤: {result['max_drawdown']*100:.2f}%")
    if 'bench_annual_return' in result:
        print(f"  基准(沪深300)年化: {result['bench_annual_return']*100:.2f}%  "
              f"超额: {result['excess_annual']*100:+.2f}pp")
    print(f"  总期数: {result['n_periods']}  危机期: {result['crisis_periods']}")
    if delist_info:
        hits = extra_info.get('delist_hits', 0)
        print(f"  退市计提: {hits} 次（按 {delist_haircut:.0%}/次）")
        result['delist_hits'] = hits
        result['delist_haircut'] = delist_haircut
    return result


def run_arc_monthly_returns(factors_df, daily_data, index_df, start_date, end_date,
                            industry_map=None):
    """
    轻量版：只收集 ARC 月度收益序列（用于相关性计算），跳过完整指标输出。
    供 eval_tid.py 调用。
    """
    factors_used, month_ends = _prepare_backtest_data(
        factors_df, daily_data, index_df, start_date, end_date, industry_map)

    factor_cols = [c for c in factors_used.columns]
    normal_weights = pd.Series({f: FACTOR_WEIGHTS.get(f, 0) for f in factor_cols})
    crisis_weights = pd.Series({f: CRISIS_WEIGHTS.get(f, 0) for f in factor_cols})

    detector = CrisisDetector(index_drawdown=CRISIS_IDX_DRAWDOWN, vol_ratio=CRISIS_VOL_RATIO,
                              exit_drawdown=CRISIS_EXIT_DRAWDOWN,
                              exit_vol_ratio=CRISIS_EXIT_VOL_RATIO,
                              exit_days=CRISIS_EXIT_DAYS,
                              fresh_start=True)

    returns_df, _, _ = run_backtest_loop(
        factors_used, daily_data, index_df, month_ends,
        normal_weights, crisis_weights, detector,
        with_cost=False)  # 相关性计算不需要扣成本

    if returns_df is None or len(returns_df) == 0:
        return pd.Series(dtype=float)

    df = returns_df.set_index('date')
    return df['ret']


def compute_market_dispersion(daily, industry_map, window=20):
    """
    计算行业动量离散度，作为市场状态识别器。

    逻辑：
    1. 按日按行业聚合等权日收益（ret 列） → 每日取所有行业收益的横截面标准差
    2. 对标准差做 252 日滚动百分位排名
    3. 返回 Series: date (pd.Timestamp) → dispersion_percentile (0~1)，index 为 date
    """
    dates = daily.index.get_level_values('date').unique()
    dispersion_std = {}

    for dt in dates:
        try:
            sub = daily.xs(dt, level='date')
        except Exception:
            continue
        if 'ret' not in sub.columns:
            continue
        sub = sub.copy()
        sub['ind'] = sub.index.map(industry_map)
        # 过滤掉无行业映射的股票
        sub = sub.dropna(subset=['ind'])
        if sub.empty or sub['ind'].nunique() < 3:
            continue
        ind_mean = sub.groupby('ind')['ret'].mean()
        dispersion_std[dt] = ind_mean.std()

    if not dispersion_std:
        return pd.Series(dtype=float)

    std_series = pd.Series(dispersion_std).sort_index()

    # 252 日滚动百分位排名
    percentile = std_series.rolling(window=252, min_periods=63).rank(pct=True)

    return percentile


def dynamic_lambda(dispersion_pct, max_lambda=0.30):
    """
    根据市场离散度百分位计算动态 λ。

    规则：
    - p < 0.3  → λ = 0
    - 0.3 ≤ p < 0.7 → λ = 0
    - p ≥ 0.7  → λ = max_lambda * (p - 0.7) / 0.3

    参数 dispersion_pct 可以为标量或 Series/array。
    """
    p = np.asarray(dispersion_pct, dtype=float)
    lam = np.where(
        p >= 0.7,
        max_lambda * (p - 0.7) / 0.3,
        0.0
    )
    if np.isscalar(dispersion_pct):
        return float(lam)
    return lam


def compute_sect_momentum_percentile(daily, industry_map):
    """
    计算每只股票所在行业的20日收益排名百分位（0~1）。

    供 C* 条件因子增强和动态 λ 回测使用。
    返回 Series(index=date,code) 的行业动量百分位。
    """
    dates = daily.index.get_level_values('date').unique()
    industry_ret = {}

    for dt in dates:
        sub = daily.xs(dt, level='date')
        if 'ret' not in sub.columns or 'close' not in sub.columns:
            continue
        sub = sub.copy()
        sub['ind'] = sub.index.map(industry_map)
        ind_mean = sub.groupby('ind')['ret'].mean()
        industry_ret[dt] = ind_mean

    if not industry_ret:
        return pd.Series(np.nan, index=daily.index)

    ind_ret_df = pd.DataFrame(industry_ret).T  # date × industry
    ind_ret_20 = ind_ret_df.rolling(20).sum()

    percentiles = {}
    for dt in ind_ret_20.index:
        row = ind_ret_20.loc[dt].dropna()
        if len(row) < 3:
            continue
        percentiles[dt] = row.rank(pct=True)

    # 映射回每只股票 — 向量化 groupby+map，替代 O(N²) 嵌套循环
    # 构建 (date, code) → industry 的 Series
    code_to_ind = pd.Series({c: industry_map.get(c)
                             for c in daily.index.get_level_values('code').unique()})
    daily_ind = code_to_ind.reindex(daily.index.get_level_values('code'))
    daily_ind.index = daily.index

    result = pd.Series(np.nan, index=daily.index)
    for dt in percentiles:
        if dt in result.index.get_level_values('date'):
            pct = percentiles[dt]
            mask = result.index.get_level_values('date') == dt
            result.loc[mask] = daily_ind.loc[mask].map(pct).values
    return result


def make_cstar_modifier(sect_pct, lambda_=0.12, alpha=0.12):
    """C* 条件因子增强：创建 score_modifier 闭包。

    规则：k = 1 + λ × tanh(α × (percentile - 0.5))
    返回 (modifier, tilt_stats)，其中 tilt_stats 会在回测过程中被填充。
    """
    tilt_stats = []

    def cstar_modifier(score, f_slice, dt, current_mode, extra_info):
        if current_mode != 'normal':
            return score
        if dt not in sect_pct.index.get_level_values('date'):
            return score
        sect_pct_date = sect_pct.xs(dt, level='date')
        common = score.index.intersection(sect_pct_date.index)
        if len(common) == 0:
            return score
        pct_vals = sect_pct_date.loc[common].clip(0, 1)
        k = 1 + lambda_ * np.tanh(alpha * (pct_vals - 0.5))
        tilt_stats.append({'date': dt, 'k_mean': k.mean(), 'k_std': k.std()})
        score.loc[common] = score.loc[common] * k.values
        return score

    return cstar_modifier, tilt_stats


def make_dynamic_lambda_modifier(sect_pct, market_dispersion, max_lambda=0.30, alpha=0.12):
    """动态 λ 行业动量修正：创建 score_modifier 闭包。

    仅在趋势市（高离散度）激活 TID 调节。
    返回 (modifier, tilt_stats)，其中 tilt_stats 会在回测过程中被填充。
    """
    tilt_stats = []

    def dynamic_lambda_modifier(score, f_slice, dt, current_mode, extra_info):
        if current_mode != 'normal':
            return score
        if dt not in sect_pct.index.get_level_values('date'):
            return score
        sect_pct_date = sect_pct.xs(dt, level='date')
        common = score.index.intersection(sect_pct_date.index)
        if len(common) == 0:
            return score

        pct_vals = sect_pct_date.loc[common].clip(0, 1)

        if dt in market_dispersion.index:
            disp_pct = market_dispersion.loc[dt]
            lam = dynamic_lambda(disp_pct, max_lambda=max_lambda) if not pd.isna(disp_pct) else 0.0
        else:
            lam = 0.0

        k = 1 + lam * np.tanh(alpha * (pct_vals - 0.5))
        tilt_stats.append({'date': dt, 'lambda': lam, 'k_mean': k.mean(), 'k_std': k.std()})
        score.loc[common] = score.loc[common] * k.values
        return score

    return dynamic_lambda_modifier, tilt_stats


