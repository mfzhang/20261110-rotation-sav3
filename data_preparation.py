#!/usr/bin/env python3
# /tmp/sav3/data_preparation.py
# 数据清洗准备 — 适配 tushare 数据格式
import pandas as pd
import numpy as np
from config import *
# 财务数据总开关（star import 已带入，显式再导入以便检索/静态检查）
from config import FINANCIAL_DATA_ENABLED

# 各板块涨跌停幅度阈值（%，留 0.2pp 余量）
# 主板 ±10%，创业板/科创板 ±20%，北交所 ±30%，ST(主板) ±5%
# 创业板 2020-08-24 注册制改革前为 ±10%
LIMIT_MAIN = 9.8
LIMIT_GROWTH = 19.8   # 创业板(300/301/302) + 科创板(688/689)
LIMIT_BJ = 29.8       # 北交所(.BJ)
LIMIT_ST = 4.8        # ST 主板（创业板/科创板 ST 仍按 20%）
CYB_REFORM_DATE = pd.Timestamp('2020-08-24')


def _price_limit_threshold(codes: pd.Series, names: pd.Series, dates: pd.Series) -> pd.Series:
    """按板块/日期返回每只股票的涨跌停阈值（%）。"""
    code = codes.astype(str)
    num = code.str[:6]
    is_kcb = num.str.startswith(('688', '689'))
    is_cyb = num.str.startswith(('300', '301', '302', '303'))
    is_bj = code.str.endswith('.BJ')
    is_st = names.astype(str).str.contains('ST', na=False)

    thr = pd.Series(LIMIT_MAIN, index=codes.index, dtype=float)
    thr[is_kcb | is_cyb] = LIMIT_GROWTH
    thr[is_bj] = LIMIT_BJ
    # 创业板注册制改革前 ±10%
    thr[is_cyb & (dates < CYB_REFORM_DATE)] = LIMIT_MAIN
    # ST 主板 ±5%（创业/科创/北交 ST 维持板块阈值）
    thr[is_st & ~(is_kcb | is_cyb | is_bj)] = LIMIT_ST
    return thr


def prepare_data(start_date, end_date, stock_limit=None, skip_financials=False):
    from data_fetcher import get_all_codes, get_stock_basic, fetch_stock_daily, fetch_index_daily

    codes = get_all_codes(limit=stock_limit)
    print(f"  股票池: {len(codes)} 只")
    start_dt = pd.to_datetime(start_date)

    # 拉取数据
    daily = fetch_stock_daily(codes, start_date, end_date)
    basic = get_stock_basic()

    # 行业映射: ts_code → industry
    # WARNING: 行业分类为静态快照（来自最新 stock_basic），存在前视偏差风险
    # 所有历史日期均使用最新的行业归属做中性化，不代表各时点真实行业分类
    industry_map = basic.set_index('ts_code')['industry']
    code_to_name = basic.set_index('ts_code')['name']

    # 上市日期 (兼容不同字段名)
    list_col = 'list_date' if 'list_date' in basic.columns else 'list_date'.upper() if 'LIST_DATE' in basic.columns else None
    if list_col:
        code_to_list_date = pd.to_datetime(basic.set_index('ts_code')[list_col])
    else:
        code_to_list_date = pd.Series(pd.Timestamp('2000-01-01'), index=basic['ts_code'])

    # 指数
    index_df = fetch_index_daily(start_date, end_date)
    index_ret = index_df['close'].pct_change().fillna(0)

    # 数据清洗
    daily = daily.reset_index()
    daily['name'] = daily['code'].map(code_to_name)

    # 过滤 ST
    daily = daily[~daily['name'].str.contains('ST', na=False)]

    # 过滤上市时间不足
    daily['list_date'] = daily['code'].map(code_to_list_date)
    daily['list_days'] = (daily['date'] - daily['list_date']).dt.days
    daily = daily[daily['list_days'] >= MIN_LIST_DAYS]

    # 涨跌停标记（STOP_FILTER）：按板块分档阈值打 tradable 标记而非删行——
    # 删行会让 pct_change 跨缺口产生伪单日收益，污染波动率/beta/IC/回测（review §3.2）。
    # 一字涨跌停当日无法成交：选股/回测买入时排除 tradable=False，价格序列保持连续。
    if STOP_FILTER:
        thr = _price_limit_threshold(daily['code'], daily['name'], daily['date'])
        daily['tradable'] = (daily['pct_chg'] < thr) & (daily['pct_chg'] > -thr)
    else:
        daily['tradable'] = True

    daily = daily.sort_values(['code', 'date'])

    # 流动性过滤：相对分位数 + 绝对成交额门槛
    if 'amount' in daily.columns:
        daily['avg_amount'] = daily.groupby('code')['amount'].transform(
            lambda x: x.rolling(LIQUIDITY_WINDOW, min_periods=1).mean())
        cutoff_rel = daily.groupby('date')['avg_amount'].transform(
            lambda x: x.quantile(LIQUIDITY_PERCENTILE))
        # 双重过滤：低于分位数 OR 低于绝对门槛的都剔除
        daily = daily[(daily['avg_amount'] >= cutoff_rel) &
                       (daily['avg_amount'] >= LIQUIDITY_MIN_AMOUNT)]

    daily = daily.set_index(['date', 'code']).sort_index()

    # 衍生指标
    daily['ret'] = daily.groupby('code')['close'].pct_change()

    # 52周高低点
    daily['high_52w'] = daily.groupby('code')['high'].transform(
        lambda x: x.rolling(HIGH_LOW_WINDOW, min_periods=1).max())
    daily['low_52w'] = daily.groupby('code')['low'].transform(
        lambda x: x.rolling(HIGH_LOW_WINDOW, min_periods=1).min())

    # 20日收益
    daily['ret_20d'] = daily.groupby('code')['close'].transform(
        lambda x: x.pct_change(periods=REL_STRENGTH_DAYS))

    # 指数收益映射
    daily['index_ret'] = daily.index.get_level_values('date').map(index_ret)

    # 财务数据：总开关未启用前，--full / skip_financials=False 一律降级为跳过
    # （tushare 财务接口尚未接入，下方 else 分支只是 TODO 占位，不会产生真实财务因子）
    if not FINANCIAL_DATA_ENABLED and not skip_financials:
        print("  财务数据未启用 (FINANCIAL_DATA_ENABLED=False)，降级为 quick 模式")
        skip_financials = True

    # 财务数据（skip 模式）
    if skip_financials:
        print("  跳过财务数据 (quick 模式)")
        daily['np_margin'] = np.nan
        daily['cash_to_np'] = np.nan
        daily['leverage'] = np.nan
    else:
        daily['np_margin'] = np.nan
        daily['cash_to_np'] = np.nan
        daily['leverage'] = np.nan
        # TODO: tushare 财务数据接口 (需要 pro.income, pro.balancesheet, pro.cashflow)
        print("  [TODO] 财务数据需接入 tushare income/balancesheet/cashflow")

    return daily, index_ret, industry_map, index_df
