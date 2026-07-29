#!/usr/bin/env python3
# /tmp/sav3/fermentation.py
"""
低位发酵检测器 — TID 策略翻转核心模块

从"推动量最强"翻转为"找低位异动"：
1. 基于日线数据构建子方向指数（成交额加权）
2. 计算 5 个拥挤度指标 + 位置 + 量价关系
3. 筛选低位发酵候选
4. A/B/C 规则分档
5. LLM 文本增强（确认条件 + 失效条件）

纯日线数据可算，无额外 API 调用。
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

from sub_sectors import (
    SUB_TO_SHENWAN, SUB_KEYWORDS, STOCK_TO_SUB_KEYWORDS,
    get_all_sub_sectors, classify_news_to_sub, classify_stock_to_sub
)
from sector_rotation import volume_weighted_return

logger = logging.getLogger("fermentation")

# 向后兼容：旧代码可能引用 fermentation._classify_stock_to_sub
_classify_stock_to_sub = classify_stock_to_sub

# === 可调参数 ===
FERMENT_POSITION_MIN = 0.50      # 52周位置必须 >=0.50（即处于52周相对低位区间）
FERMENT_VOL_RATIO_MIN = 1.0      # 量比必须 >1.0（有放量）
FERMENT_CONSECUTIVE_MAX = 5      # 连阳天数 <5（还没连续大涨）
FERMENT_CROWDING_THRESHOLD = 0.70  # 拥挤度 >0.7 直接排除
FERMENT_LOOKBACK = 252            # 52周窗口（交易日）


def _build_sub_indices(daily_df: pd.DataFrame, industry_map: pd.Series) -> dict:
    """
    从 daily_df 构建子方向指数。

    返回: {sub_name: (ret_series, amt_series, n_stocks)}
      ret_series: pd.Series[date → daily_return(decimal)]
      amt_series: pd.Series[date → total_amount]
      n_stocks: int — 该子方向实际包含的股票数
    """
    # 取 252+5 个交易日
    all_dates = sorted(daily_df.index.get_level_values('date').unique())
    all_dates = pd.DatetimeIndex(all_dates)  # 确保是 DatetimeIndex
    if len(all_dates) < FERMENT_LOOKBACK:
        logger.warning(f"日线数据不足 {len(all_dates)} < {FERMENT_LOOKBACK}")
        return {}

    recent_dates = all_dates[-FERMENT_LOOKBACK-5:]
    daily = daily_df.loc[pd.IndexSlice[recent_dates, :]].copy()

    # 展平 + pivot
    daily_flat = daily.reset_index()
    ret_matrix = daily_flat.pivot(index='date', columns='code', values='pct_chg')
    amt_matrix = daily_flat.pivot(index='date', columns='code', values='amount')

    # 分类每只股票到子方向（优先用 SUB_KEYWORDS 关键词匹配，industry_map 仅作 fallback）
    sub_codes = {}
    for code in ret_matrix.columns:
        assigned = False
        # 优先关键词匹配到子方向
        name = ""
        try:
            name_row = daily_flat[daily_flat['code'] == code]
            if 'name' in name_row.columns:
                name = str(name_row['name'].iloc[0]) if not name_row.empty else ""
        except Exception:
            pass
        sub = classify_stock_to_sub(str(code), name)
        if sub is not None:
            sub_codes.setdefault(sub, []).append(code)
            assigned = True
        # fallback: industry_map 申万一级行业（仅在关键词匹配失败时）
        if not assigned and code in industry_map.index:
            sw = str(industry_map.loc[code])
            if sw and sw not in ('nan', 'None', ''):
                sub_codes.setdefault(sw, []).append(code)

    # 按子方向聚合
    sub_indices = {}
    for sub_name, codes in sub_codes.items():
        valid_codes = [c for c in codes if c in ret_matrix.columns and c in amt_matrix.columns]
        if len(valid_codes) < 3:
            continue
        daily_mean = volume_weighted_return(ret_matrix, amt_matrix, valid_codes,
                                            min_codes=3, ret_in_percent=True)
        if len(daily_mean) >= 20:
            sub_amt = amt_matrix[valid_codes].clip(lower=0)
            amt_sum = sub_amt.sum(axis=1).loc[daily_mean.index]
            # 真实成分股广度：每日上涨家数占比（替代"指数收益符号"伪广度，review §4.4）
            sub_ret = ret_matrix[valid_codes]
            breadth = (sub_ret > 0).sum(axis=1) / sub_ret.notna().sum(axis=1).replace(0, np.nan)
            breadth = breadth.loc[daily_mean.index]
            sub_indices[sub_name] = (daily_mean, amt_sum, len(valid_codes), breadth)

    logger.info(f"子方向指数: {len(sub_indices)} 个")
    return sub_indices


def compute_crowding_metrics(sub_indices: dict) -> pd.DataFrame:
    """
    计算每个子方向的拥挤度相关指标。

    返回 DataFrame(子方向, 列见下):
    """
    rows = []
    for sub_name, item in sub_indices.items():
        # 兼容 3 元组（无广度）与 4 元组（含真实成分股广度）
        ret_s, amt_s, n_stocks_val = item[0], item[1], item[2]
        breadth_s = item[3] if len(item) > 3 else None
        if len(ret_s) < 20:
            continue

        # 对齐 ret_s 和 amt_s 的日期索引（避免索引不一致导致计算偏差）
        common_idx = ret_s.index.intersection(amt_s.index)
        ret_s = ret_s.loc[common_idx]
        amt_s = amt_s.loc[common_idx]
        if len(ret_s) < 20:
            continue

        # 近20日收益率
        ret_20d = ret_s.iloc[-20:]
        cum_ret = (1 + ret_20d).prod() - 1

        # 量比：5日均量 / 20日均量
        vol_5d = amt_s.iloc[-5:].mean()
        vol_20d = amt_s.iloc[-20:].mean()
        vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1.0

        # 连阳天数
        consecutive_up = 0
        for r in reversed(ret_s.values):
            if r > 0:
                consecutive_up += 1
            else:
                break

        # 广度：近3/10日成分股上涨家数占比均值（真实广度）
        # 无广度数据时降级为"指数收益为正日期占比"（伪广度，仅 4 个离散值）
        if breadth_s is not None and len(breadth_s.dropna()) >= 10:
            breadth_3d = float(breadth_s.iloc[-3:].mean())
            breadth_10d = float(breadth_s.iloc[-10:].mean())
        else:
            breadth_3d = (ret_s.iloc[-3:] > 0).mean()
            breadth_10d = (ret_s.iloc[-10:] > 0).mean()
        breadth_momentum = breadth_3d - breadth_10d

        # 成交额分位
        amt_252 = amt_s.iloc[-252:] if len(amt_s) >= 252 else amt_s
        amt_latest = amt_s.iloc[-1] if len(amt_s) > 0 else 0
        turnover_rank_pct = (amt_252 < amt_latest).mean()

        # 涨跌比（正收益日 / 负收益日，近20日）
        up_days = (ret_20d > 0).sum()
        down_days = (ret_20d < 0).sum()
        up_down_ratio = up_days / down_days if down_days > 0 else up_days

        # 52周位置
        cum_all = (1 + ret_s).cumprod()
        high_52w = cum_all.iloc[-252:].max() if len(cum_all) >= 252 else cum_all.max()
        low_52w = cum_all.iloc[-252:].min() if len(cum_all) >= 252 else cum_all.min()
        current = cum_all.iloc[-1]
        position = (high_52w - current) / (high_52w - low_52w) if high_52w != low_52w else 0.5

        # 拥挤度综合得分
        crowding_score = (
            0.4 * min(consecutive_up / 10, 1.0)
            + 0.3 * turnover_rank_pct
            + 0.3 * (1.0 if cum_ret > 0.15 else cum_ret / 0.15 if cum_ret > 0 else 0)
        )

        rows.append({
            'sub_sector': sub_name,
            'shenwan': SUB_TO_SHENWAN.get(sub_name, sub_name),
            'return_20d': cum_ret,
            'volume_ratio': vol_ratio,
            'consecutive_up': consecutive_up,
            'breadth_3d': breadth_3d,
            'breadth_momentum': breadth_momentum,
            'turnover_rank_pct': turnover_rank_pct,
            'up_down_ratio': up_down_ratio,
            'position': position,
            'crowding_score': crowding_score,
            'n_stocks': n_stocks_val,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def find_low_fermentation(sub_indices: dict, crowding_df: pd.DataFrame) -> pd.DataFrame:
    """
    筛选低位发酵候选 + A/B/C 分档。

    返回 DataFrame 包含分档结果。
    """
    if crowding_df.empty:
        return pd.DataFrame()

    df = crowding_df.copy()

    # === Phase 3: 候选筛选 ===
    # 排除已拥挤
    df = df[df['crowding_score'] <= FERMENT_CROWDING_THRESHOLD].copy()
    # 位置不够低
    df = df[df['position'] >= FERMENT_POSITION_MIN].copy()
    # 没有放量
    df = df[df['volume_ratio'] >= FERMENT_VOL_RATIO_MIN].copy()
    # 已经连涨太多天
    df = df[df['consecutive_up'] < FERMENT_CONSECUTIVE_MAX].copy()

    if df.empty:
        return pd.DataFrame()

    # 发酵得分
    # 权重再平衡（2026-07-19 模型层修复）：position 0.35→0.15，
    # 差额分配给量比(+0.10)和广度动量(+0.10)。
    # 原权重下 position 一家独大，"跌得多"即可高分，量价确认形同虚设；
    # 新权重让放量与广度扩散成为一等信号，position 只做低位守门。
    df['ferment_score'] = (
        0.15 * df['position']
        + 0.35 * np.minimum(df['volume_ratio'], 2.0) / 2.0
        + 0.30 * np.clip(np.maximum(df['breadth_momentum'] + 0.1, 0), 0, 1)
        + 0.10 * df['breadth_3d']
        + 0.10 * (1 - df['return_20d'].clip(0, 0.3) / 0.3)
    )

    df = df.sort_values('ferment_score', ascending=False).reset_index(drop=True)

    # === Phase 4: A/B/C 规则分档 ===
    tiers = []
    for _, row in df.iterrows():
        score = row['ferment_score']
        vol_ratio = row['volume_ratio']
        if score >= 0.70 and vol_ratio >= 1.3:
            tiers.append('A')
        elif score >= 0.55 and vol_ratio >= 1.3:
            tiers.append('B')
        elif score >= 0.40:
            tiers.append('C')
        else:
            tiers.append('D')

    df['tier'] = tiers
    # 只保留 A/B/C
    df = df[df['tier'].isin(['A', 'B', 'C'])].reset_index(drop=True)

    logger.info(f"低位发酵候选: A={len(df[df['tier']=='A'])}, "
                f"B={len(df[df['tier']=='B'])}, C={len(df[df['tier']=='C'])}")
    return df


def run_fermentation_pipeline(daily_df: pd.DataFrame, industry_map: pd.Series) -> dict:
    """
    完整低位发酵管线。

    输入: ARC 共享的 daily_df + industry_map
    输出: {
        'crowding': DataFrame (所有子方向拥挤度指标),
        'fermentation': DataFrame (A/B/C 分档结果),
        'sub_indices': dict (子方向指数),
        'excluded_crowded': list (被排除的拥挤子方向),
    }
    """
    # Phase 1-2: 建指数 + 特征
    sub_indices = _build_sub_indices(daily_df, industry_map)
    if not sub_indices:
        return {
            'crowding': pd.DataFrame(),
            'fermentation': pd.DataFrame(),
            'sub_indices': {},
            'excluded_crowded': [],
        }

    crowding = compute_crowding_metrics(sub_indices)

    # 被排除的拥挤方向
    excluded = []
    if not crowding.empty:
        crowded_mask = crowding['crowding_score'] > FERMENT_CROWDING_THRESHOLD
        excluded = crowding[crowded_mask][['sub_sector', 'crowding_score',
                                            'return_20d']].to_dict('records')

    # Phase 3-4: 发酵筛选 + 分档
    fermentation = find_low_fermentation(sub_indices, crowding)

    return {
        'crowding': crowding,
        'fermentation': fermentation,
        'sub_indices': sub_indices,
        'excluded_crowded': excluded,
    }


# === LLM 增强 ===
def generate_fermentation_llm_text(fermentation_df: pd.DataFrame,
                                   news_items: list = None,
                                   target_tiers: list = None) -> dict:
    """
    对 A/B 档生成 LLM 增强文本（确认条件 + 失效条件）。

    返回: {sub_sector: {'catalyst': str, 'confirm_condition': str, 'fail_condition': str}}

    注意：此函数准备 LLM prompt 和调用逻辑，实际 API 调用由 main.py 处理。
    这里返回原始数据供 main.py 组装 prompt。
    """
    if target_tiers is None:
        target_tiers = ['A', 'B']

    prompt_data = {}
    for _, row in fermentation_df.iterrows():
        if row['tier'] not in target_tiers:
            continue
        sub = row['sub_sector']
        prompt_data[sub] = {
            'sub_sector': sub,
            'shenwan': row['shenwan'],
            'position': row['position'],
            'volume_ratio': row['volume_ratio'],
            'breadth_3d': row['breadth_3d'],
            'breadth_momentum': row['breadth_momentum'],
            'return_20d': row['return_20d'],
            'tier': row['tier'],
            'news': [],  # 由 main.py 注入
        }

    return prompt_data


if __name__ == '__main__':
    # 快速自检
    print(f"子方向数: {len(get_all_sub_sectors())}")
    from sub_sectors import SHENWAN_TO_SUB
    for sw, subs in SHENWAN_TO_SUB.items():
        print(f"  {sw}: {len(subs)} 子方向")
