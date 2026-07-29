#!/usr/bin/env python3
# key_levels.py
# 关键点位计算

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("market_timing")


class KeyLevelsCalculator:
    """关键点位计算"""

    def __init__(self, parent):
        self.parent = parent

    def compute_bollinger(self, close: pd.Series, window: int = 20, n_std: float = 2.0
                          ) -> Dict[str, pd.Series]:
        """布林带"""
        ma = close.rolling(window).mean()
        std = close.rolling(window).std()
        return {
            'upper': ma + n_std * std,
            'middle': ma,
            'lower': ma - n_std * std,
            'width': (ma + n_std * std - (ma - n_std * std)) / ma,  # 带宽
        }

    def compute_key_levels(self) -> Dict[str, Dict]:
        """计算各指数的关键支撑/压力位"""
        close_df = self.parent.prepare_data()[1]
        result = {}

        for name in close_df.columns:
            s = close_df[name].dropna()
            if len(s) < 250:
                continue

            current = s.iloc[-1]

            levels = {
                'current_close': round(float(current), 2),
            }

            # ── 均线 ──
            for ma_n in [5, 10, 20, 60, 120, 250]:
                ma = s.rolling(ma_n).mean().iloc[-1]
                if not pd.isna(ma):
                    dist = (current - ma) / ma
                    levels[f'MA{ma_n}'] = round(float(ma), 2)
                    levels[f'MA{ma_n}_dist'] = round(float(dist), 4)

            # ── 布林带 (20,2) ──
            bb = self.compute_bollinger(s, 20, 2.0)
            for band in ['upper', 'middle', 'lower']:
                val = bb[band].iloc[-1]
                if not pd.isna(val):
                    levels[f'BB_{band}'] = round(float(val), 2)
            bw = bb['width'].iloc[-1]
            if not pd.isna(bw):
                levels['BB_width'] = round(float(bw), 4)

            # ── 近期高低点 ──
            for window, label in [(20, '20d'), (60, '60d'), (120, '120d'), (250, '250d')]:
                if len(s) >= window:
                    hh = s.iloc[-window:].max()
                    ll = s.iloc[-window:].min()
                    levels[f'{label}_high'] = round(float(hh), 2)
                    levels[f'{label}_low'] = round(float(ll), 2)
                    levels[f'{label}_high_dist'] = round(float((current - hh) / hh), 4)
                    levels[f'{label}_low_dist'] = round(float((current - ll) / ll), 4)

            # ── 前高前低 (swing points) ──
            swings = self._find_swing_points(s)
            resistances = [p for p in swings if p > current]
            supports = [p for p in swings if p < current]
            levels['swing_resistance'] = sorted(resistances, reverse=True)[:3] if resistances else []
            levels['swing_support'] = sorted(supports)[:3] if supports else []

            # ── 斐波那契回撤 (基于最近一段大趋势) ──
            fib = self._compute_fibonacci(s)
            levels.update(fib)

            # ── 综合压力/支撑 ──
            all_resistance = [current * 1.0]  # 最小压力位
            for k in ['BB_upper', 'MA20', 'MA60', '60d_high', '120d_high']:
                if k in levels:
                    all_resistance.append(levels[k])
            for r in levels.get('swing_resistance', []):
                all_resistance.append(r)
            # 取高于当前价的最近5个
            above = sorted([r for r in all_resistance if r > current])
            levels['key_resistance'] = above[:5] if above else []

            all_support = [current * 1.0]
            for k in ['BB_lower', 'MA20', 'MA60', '20d_low', '60d_low', '120d_low']:
                if k in levels:
                    all_support.append(levels[k])
            for s_ in levels.get('swing_support', []):
                all_support.append(s_)
            below = sorted([s_ for s_ in all_support if s_ < current], reverse=True)
            levels['key_support'] = below[:5] if below else []

            # ── 原型模型中提到的关键点位 (当天数据) ──
            levels['prototype_jingxian'] = self._estimate_jingxian(name, s)

            result[name] = levels

        return result

    def _find_swing_points(self, series: pd.Series, window: int = 10) -> List[float]:
        """找摆动高低点 (局部极值)"""
        points = []
        for i in range(window, len(series) - window):
            left = series.iloc[i - window:i]
            right = series.iloc[i + 1:i + window + 1]
            val = series.iloc[i]
            if val > left.max() and val > right.max():
                points.append(float(val))  # 局部高点
            elif val < left.min() and val < right.min():
                points.append(float(val))  # 局部低点
        return points

    def _compute_fibonacci(self, series: pd.Series, lookback: int = 250) -> Dict[str, float]:
        """计算基于近期大趋势的斐波那契回撤位"""
        if len(series) < lookback:
            lookback = len(series)

        segment = series.iloc[-lookback:]
        high = segment.max()
        low = segment.min()
        high_idx = segment.idxmax()
        low_idx = segment.idxmin()

        # 判断趋势方向
        if high_idx > low_idx:
            # 上升趋势: 回撤
            direction = 'uptrend'
        else:
            # 下降趋势: 反弹
            direction = 'downtrend'

        diff = high - low
        if diff == 0:
            return {}

        fib_levels = {}
        for ratio, name in [(0.236, 'fib_236'), (0.382, 'fib_382'),
                             (0.5, 'fib_500'), (0.618, 'fib_618'),
                             (0.786, 'fib_786'), (1.0, 'fib_100'),
                             (1.272, 'fib_1272'), (1.618, 'fib_1618')]:
            if direction == 'uptrend':
                level = high - diff * ratio
            else:
                # 下降趋势反弹: retrace = base - (base - low) * fib  →  low + diff * ratio
                level = low + diff * ratio
            fib_levels[name] = round(float(level), 2)

        fib_levels['fib_direction'] = direction
        fib_levels['fib_high'] = round(float(high), 2)
        fib_levels['fib_low'] = round(float(low), 2)
        return fib_levels

    def _estimate_jingxian(self, name: str, series: pd.Series) -> Optional[float]:
        """估算颈线位 (基于近期高点平台)"""
        if len(series) < 60:
            return None

        recent = series.iloc[-60:]
        # 颈线约在近期20日高点下方2-3%
        hh_20 = recent.iloc[-20:].max()
        jingxian = hh_20 * 0.975
        return round(float(jingxian), 2)
