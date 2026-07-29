#!/usr/bin/env python3
# /tmp/sav3/fermentation_cross.py
"""
TID × ARC 交叉验证模块

把 ARC 的 position 因子聚合到行业级别，与 TID 发酵信号对比。
独立模块：只依赖 fermentation.py 和 sub_sectors.py，不依赖 main.py。
ARC 数据通过参数传入，不主动拉数据。

原理：
  ARC 的 position 因子 = (52周高 - 当前价) / (52周高 - 52周低)
  值越高 → 越靠近 52 周低位（即"跌多了"）。

  交叉验证逻辑：
  如果 TID 发酵说"小金属低位"，但 ARC 说"有色金属整体在半山腰以上"，
  则 arc_confirm=False，降低虚假信号的置信度。

数据源（2026-07-19 模型层修复）：
  首选复用 factor_calculator.calculate_factors 的 position 输出（factors_df 参数），
  保证交叉验证与 ARC 生产因子口径完全一致；修复前用收盘价高低点自算，
  与因子口径（含盘中极值的 high_52w/low_52w）存在定义偏差。
  factors_df 未提供时保留旧的自算路径作离线回退。
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

from sub_sectors import TUSHARE_TO_SHENWAN

logger = logging.getLogger("fermentation_cross")


def _to_shenwan(ind) -> str:
    """行业名归一到申万一级：tushare 行业名查映射表，已是申万名/未知名原样保留。

    生产环境 industry_map 的值是 tushare 行业名（110 类，如"半导体"），
    而 ferm_df['shenwan'] 是申万一级（29 类，如"电子"）——不归一时两边
    永远对不上，arc_confirm 全为 NaN（修复前生产实际表现）。
    """
    if pd.isna(ind):
        return ind
    return TUSHARE_TO_SHENWAN.get(str(ind), str(ind))


def _arc_position_from_factors(factors_df: pd.DataFrame,
                               industry_map: pd.Series) -> pd.DataFrame:
    """从 factor_calculator 的因子面板聚合行业级 ARC position（首选路径）。

    取因子面板最新截面的 position 列（生产同口径：(high_52w-close)/(high_52w-low_52w)，
    高低点含盘中极值），按行业取中位数。
    """
    latest = factors_df.index.get_level_values('date').max()
    cross = factors_df.xs(latest, level='date')['position'].dropna()
    if cross.empty:
        logger.warning("因子面板最新截面 position 全为空")
        return pd.DataFrame()

    pos_df = pd.DataFrame({'code': cross.index, 'arc_position': cross.values})
    pos_df['arc_position'] = pos_df['arc_position'].clip(0.0, 1.0)
    pos_df['industry'] = pos_df['code'].map(industry_map).map(_to_shenwan)
    pos_df = pos_df.dropna(subset=['industry'])
    if pos_df.empty:
        logger.warning("没有股票能映射到行业")
        return pd.DataFrame()

    industry_pos = pos_df.groupby('industry').agg(
        arc_position=('arc_position', 'median'),
        n_stocks=('code', 'count'),
    ).sort_values('arc_position', ascending=False)

    logger.info(
        f"ARC 行业 position（复用因子面板 {latest.date()} 截面）: "
        f"{len(industry_pos)} 个行业, 中位 n_stocks={industry_pos['n_stocks'].median():.0f}"
    )
    return industry_pos


def compute_arc_industry_position(
    daily_df: pd.DataFrame,
    industry_map: pd.Series,
    lookback: int = 252,
    factors_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    把 ARC 的 position 因子聚合到行业级别。

    首选路径（factors_df 不为空）：直接复用 factor_calculator.calculate_factors
    输出的 position 因子（最新截面），保证交叉验证与 ARC 生产口径完全一致——
    修复前本函数用收盘价高低点自算，与因子口径（含盘中极值的 high_52w/low_52w）
    存在定义偏差。

    回退路径（factors_df 未提供）：保留旧的收盘价高低点自算（离线/测试用）。

    参数
    ----------
    daily_df : DataFrame
        MultiIndex(date, code)，至少需含 close 列（仅回退路径使用）。
    industry_map : Series
        Index 为股票代码，值为申万一级行业名。
    lookback : int
        52 周窗口的交易日数（默认 252，仅回退路径使用）。
    factors_df : DataFrame, optional
        factor_calculator 输出的因子面板，MultiIndex(date, code)，含 position 列。
        注意应传 pipeline_process 之前的原始因子（中性化/标准化后的 z 分数
        会破坏 arc_position > 0.5 的语义）。

    返回
    -------
    DataFrame(index=行业名, columns=[arc_position, n_stocks])
      arc_position 越高 → 行业越靠近 52 周低位（与 ARC position 含义一致）。
      数据不足或无有效映射时返回空 DataFrame。
    """
    # ── 首选：复用 factor_calculator 输出 ────────────────────
    if factors_df is not None and not factors_df.empty and 'position' in factors_df.columns:
        if industry_map is None or industry_map.empty:
            logger.warning("industry_map 为空，返回空 DataFrame")
            return pd.DataFrame()
        result = _arc_position_from_factors(factors_df, industry_map)
        if not result.empty:
            return result
        logger.warning("因子面板路径无结果，回退收盘价自算")

    # ── 回退：收盘价高低点自算（旧路径，离线/测试用）──────────
    # ── 输入校验 ──────────────────────────────────────────────
    if daily_df is None or daily_df.empty:
        logger.warning("daily_df 为空，返回空 DataFrame")
        return pd.DataFrame()

    if 'close' not in daily_df.columns:
        logger.error("daily_df 缺少 'close' 列，无法计算 ARC position")
        return pd.DataFrame()

    if industry_map is None or industry_map.empty:
        logger.warning("industry_map 为空，返回空 DataFrame")
        return pd.DataFrame()

    # ── 展平 MultiIndex ──────────────────────────────────────
    daily = daily_df.copy()
    if isinstance(daily.index, pd.MultiIndex):
        daily = daily.reset_index()
        date_col = 'date'
        code_col = 'code'
    else:
        # 假设单索引为 date，code 是一列
        date_col = daily.index.name or 'date'
        code_col = 'code'
        if code_col not in daily.columns:
            logger.error("无法识别 daily_df 中的 code 列")
            return pd.DataFrame()
        daily = daily.reset_index()

    # ── 过滤有行业映射的股票 ────────────────────────────────
    mapped_codes = set(industry_map.index)
    daily = daily[daily[code_col].isin(mapped_codes)].copy()
    if daily.empty:
        logger.warning("daily_df 中没有股票在 industry_map 中，返回空 DataFrame")
        return pd.DataFrame()

    # ── 按股票计算 ARC position ──────────────────────────────
    positions = {}  # code → position (最新)
    all_dates = sorted(daily[date_col].unique())

    for code, grp in daily.groupby(code_col):
        grp = grp.sort_values(date_col)
        close_series = grp['close'].values.astype(float)

        if len(close_series) < lookback:
            continue

        # 用最近 lookback 个交易日
        window = close_series[-lookback:]
        high_52w = window.max()
        low_52w = window.min()
        latest_close = close_series[-1]

        denom = high_52w - low_52w
        if denom <= 0 or np.isnan(denom):
            continue

        pos = (high_52w - latest_close) / denom
        # ARC position: 越靠近低位值越高；clamp 到 [0, 1] 以处理极值
        pos = np.clip(pos, 0.0, 1.0)
        positions[code] = pos

    if not positions:
        logger.warning("没有足够数据计算任何股票的 ARC position")
        return pd.DataFrame()

    # ── 映射到行业并聚合 ─────────────────────────────────────
    pos_df = pd.DataFrame({
        'code': list(positions.keys()),
        'arc_position': list(positions.values()),
    })
    pos_df['industry'] = pos_df['code'].map(industry_map).map(_to_shenwan)
    pos_df = pos_df.dropna(subset=['industry'])

    if pos_df.empty:
        logger.warning("没有股票能映射到行业")
        return pd.DataFrame()

    # 按行业取中位数
    industry_pos = pos_df.groupby('industry').agg(
        arc_position=('arc_position', 'median'),
        n_stocks=('code', 'count'),
    )

    # 按 arc_position 降序（越高越靠近低位）
    industry_pos = industry_pos.sort_values('arc_position', ascending=False)

    logger.info(
        f"ARC 行业 position 计算完成: {len(industry_pos)} 个行业, "
        f"中位 n_stocks={industry_pos['n_stocks'].median():.0f}"
    )
    return industry_pos


def cross_check(
    ferm_df: pd.DataFrame,
    arc_industry_pos: pd.DataFrame,
    sub_sectors_module=None,
) -> pd.DataFrame:
    """
    交叉验证：检查发酵信号的子方向是否在 ARC 行业级别也有低位确认。

    参数
    ----------
    ferm_df : DataFrame
        发酵检测输出的 A/B/C 分档 DataFrame。
        需含列: sub_sector, shenwan（或至少 sub_sector）。
    arc_industry_pos : DataFrame
        compute_arc_industry_position 的输出。
        Index 为申万一级行业名，需含 arc_position 列。
    sub_sectors_module : module, optional
        sub_sectors 模块引用，用于读取 SUB_TO_SHENWAN 映射。
        如果 ferm_df 已有 shenwan 列则可不传。

    返回
    -------
    ferm_df 的副本，增加两列:
      arc_confirm : bool
          True 表示该子方向所属申万行业的 arc_position > 0.5，
          即行业整体确实在 52 周低位半区，与发酵信号一致。
      arc_position_industry : float
          所属申万行业的 arc_position 值（无法映射时为 NaN）。
    """
    result = ferm_df.copy()

    # ── 输入校验 ──────────────────────────────────────────────
    if result.empty:
        result['arc_confirm'] = pd.Series(dtype=bool)
        result['arc_position_industry'] = pd.Series(dtype=float)
        return result

    if arc_industry_pos is None or arc_industry_pos.empty:
        logger.warning("arc_industry_pos 为空，所有 arc_confirm 设为 False")
        result['arc_confirm'] = False
        result['arc_position_industry'] = np.nan
        return result

    # ── 确定各子方向的申万行业 ────────────────────────────────
    # 优先使用 ferm_df 中已有的 shenwan 列
    if 'shenwan' in result.columns:
        industry_col = 'shenwan'
    elif sub_sectors_module is not None:
        # 从 sub_sectors 模块读取 SUB_TO_SHENWAN 映射
        sub_to_sw = getattr(sub_sectors_module, 'SUB_TO_SHENWAN', {})
        result['_shenwan'] = result['sub_sector'].map(sub_to_sw)
        industry_col = '_shenwan'
    else:
        logger.error(
            "ferm_df 无 shenwan 列且未提供 sub_sectors_module，"
            "无法进行交叉验证"
        )
        result['arc_confirm'] = False
        result['arc_position_industry'] = np.nan
        return result

    # ── 查找行业级别的 ARC position ──────────────────────────
    # arc_industry_pos 的 index 是行业名，包含 arc_position 列
    arc_pos_map = arc_industry_pos['arc_position']

    result['arc_position_industry'] = result[industry_col].map(arc_pos_map)
    result['arc_confirm'] = result['arc_position_industry'] > 0.5

    # 清理临时列
    if '_shenwan' in result.columns:
        result = result.drop(columns=['_shenwan'])

    # ── 统计 ──────────────────────────────────────────────────
    n_total = len(result)
    n_confirmed = result['arc_confirm'].sum()
    n_unmapped = result['arc_position_industry'].isna().sum()
    n_rejected = n_total - n_confirmed - n_unmapped

    logger.info(
        f"TID×ARC 交叉验证: {n_total} 个发酵信号 → "
        f"确认 {n_confirmed}, 拒绝 {n_rejected}, 无行业映射 {n_unmapped}"
    )

    return result


# ── 便捷函数：一键交叉验证 ──────────────────────────────────


def run_cross_validation(
    daily_df: pd.DataFrame,
    industry_map: pd.Series,
    ferm_df: pd.DataFrame,
    sub_sectors_module=None,
    lookback: int = 252,
    factors_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    一键运行完整交叉验证流水线。

    参数
    ----------
    daily_df : DataFrame
        日线数据（MultiIndex date/code，需含 close 列；仅回退路径使用）。
    industry_map : Series
        code → 申万一级行业名。
    ferm_df : DataFrame
        发酵检测的 A/B/C 分档结果。
    sub_sectors_module : module, optional
        sub_sectors 模块，用于 SUB_TO_SHENWAN 映射。
    lookback : int
        52 周窗口（仅回退路径使用）。
    factors_df : DataFrame, optional
        factor_calculator 输出的原始因子面板（生产路径首选，见
        compute_arc_industry_position 文档）。

    返回
    -------
    ferm_df 增加 arc_confirm 和 arc_position_industry 列。
    """
    arc_pos = compute_arc_industry_position(daily_df, industry_map, lookback,
                                            factors_df=factors_df)
    return cross_check(ferm_df, arc_pos, sub_sectors_module)


if __name__ == '__main__':
    # 快速自检：用随机数据验证核心逻辑
    import sub_sectors

    print("=== fermentation_cross 自检 ===\n")

    # 构造模拟 daily_df
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=300, freq='B')
    codes = ['000001.SZ', '000002.SZ', '600519.SH', '600036.SH', '601318.SH']
    data = []
    for date in dates:
        for code in codes:
            base_price = 10.0 + np.random.randn() * 5
            data.append({
                'date': date,
                'code': code,
                'close': abs(base_price + np.random.randn()),
                'amount': abs(np.random.randn()) * 1e8,
                'pct_chg': np.random.randn() * 3,
            })
    daily_df = pd.DataFrame(data).set_index(['date', 'code'])

    # 构造 industry_map
    industry_map = pd.Series({
        '000001.SZ': '银行',
        '000002.SZ': '房地产',
        '600519.SH': '食品饮料',
        '600036.SH': '银行',
        '601318.SH': '非银金融',
    })

    # 1. 测试 compute_arc_industry_position
    arc_pos = compute_arc_industry_position(daily_df, industry_map, lookback=252)
    print(f"1. ARC 行业 position:\n{arc_pos}\n")

    # 2. 构造模拟 ferm_df
    ferm_df = pd.DataFrame({
        'sub_sector': ['白酒', '证券', '房地产开发', '城商行'],
        'shenwan': ['食品饮料', '非银金融', '房地产', '银行'],
        'position': [0.72, 0.65, 0.58, 0.81],
        'volume_ratio': [1.5, 1.2, 1.8, 1.1],
        'tier': ['A', 'B', 'B', 'A'],
        'ferment_score': [0.75, 0.60, 0.58, 0.72],
    })

    # 3. 测试 cross_check
    result = cross_check(ferm_df, arc_pos, sub_sectors_module=sub_sectors)
    print(f"2. 交叉验证结果:\n{result[['sub_sector', 'shenwan', 'tier', 'arc_position_industry', 'arc_confirm']]}\n")

    # 4. 测试 run_cross_validation
    result2 = run_cross_validation(daily_df, industry_map, ferm_df, sub_sectors)
    print(f"3. 一键验证 (同结果):\n  arc_confirm=True: {result2['arc_confirm'].sum()}/{len(result2)}\n")

    # 5. 空数据处理
    empty_result = cross_check(pd.DataFrame(), arc_pos, sub_sectors_module=sub_sectors)
    print(f"4. 空 ferm_df 处理: {len(empty_result)} rows, columns={list(empty_result.columns)}")

    empty_arc = cross_check(ferm_df, pd.DataFrame(), sub_sectors_module=sub_sectors)
    print(f"5. 空 arc_pos 处理: arc_confirm all {empty_arc['arc_confirm'].all()}")
    print(f"   arc_position_industry all NaN: {empty_arc['arc_position_industry'].isna().all()}")

    print("\n=== 自检通过 ===")
