#!/usr/bin/env python3
# pattern_backtester.py
# 历史回测 (严格/宽口径)

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("market_timing")


class PatternBacktester:
    """历史回测 (严格/宽口径)"""

    def __init__(self, parent):
        self.parent = parent

    def backtest_strict_criteria(self, drop_threshold: float = -0.06,
                                  recovery_threshold: float = 0.06,
                                  recovery_days: int = 8) -> Dict:
        """
        严格口径回测 (原型模型):
        - 创业板+科创50 同时急跌后强修复
        - 突破/接近60日高点
        - 成长明显强于上证

        Returns:
            dict with statistics of subsequent performance
        """
        returns, close = self.parent.prepare_data()

        if '创业板指' not in close.columns or '科创50' not in close.columns:
            return {'error': '缺少创业板指或科创50数据'}

        # 检测创业板和科创50的「急跌后强修复」模式
        cyb = close['创业板指']
        kc50 = close['科创50']
        sh = close.get('上证指数', cyb)  # 上证fallback

        # 5日急跌 (发生在8日修复之前: 13天前→8天前) + 8日强修复 (8天前→今)
        cyb_drop = (cyb.shift(recovery_days) / cyb.shift(recovery_days + 5) - 1)  # 5日涨幅, 结束于8天前
        cyb_recovery = cyb.pct_change(recovery_days)   # 最近8天
        kc_drop = (kc50.shift(recovery_days) / kc50.shift(recovery_days + 5) - 1)
        kc_recovery = kc50.pct_change(recovery_days)

        matches = []
        for i in range(recovery_days + 5, len(close) - 20):
            # 创业板急跌后强修复
            cyb_d = cyb_drop.iloc[i]
            cyb_r = cyb_recovery.iloc[i]
            kc_d = kc_drop.iloc[i]
            kc_r = kc_recovery.iloc[i]

            if pd.isna(cyb_d) or pd.isna(kc_d):
                continue
            if cyb_d > drop_threshold or kc_d > drop_threshold:
                continue  # 没有急跌
            if cyb_r < recovery_threshold or kc_r < recovery_threshold:
                continue  # 没有强修复

            # 突破/接近60日高点 (距高点 < 2%)
            cyb_60h = cyb.iloc[max(0, i - 59):i + 1].max()
            kc_60h = kc50.iloc[max(0, i - 59):i + 1].max()
            cyb_dist = (cyb.iloc[i] - cyb_60h) / cyb_60h
            kc_dist = (kc50.iloc[i] - kc_60h) / kc_60h

            if cyb_dist < -0.03 or kc_dist < -0.03:
                continue  # 离60日高点太远

            # 成长强于上证
            cyb_8d = returns['创业板指'].iloc[i - 7:i + 1].sum() if i >= 7 else 0
            sh_8d = returns['上证指数'].iloc[i - 7:i + 1].sum() if i >= 7 else 0
            excess = cyb_8d - sh_8d
            if excess < 0.03:
                continue  # 超额不够

            # 记录后续表现
            row = {'date': close.index[i], 'cyb_excess': excess}
            for n in [1, 2, 3, 5, 10, 20]:
                if i + n < len(close):
                    for name in ['创业板指', '科创50', '上证指数']:
                        if name in close.columns:
                            s = close[name]
                            fwd = (s.iloc[i + n] - s.iloc[i]) / s.iloc[i]
                            row[f'{name}_fwd{n}d'] = fwd
            matches.append(row)

        if not matches:
            return {'strict_matches': 0, 'message': '无匹配的严格口径场景'}

        df = pd.DataFrame(matches)
        result = self._summarize_backtest(df, 'strict')
        result.update(self._unconditional_baseline(cyb))
        return result

    def backtest_wide_criteria(self, drop_threshold: float = -0.05,
                                recovery_threshold: float = 0.05,
                                recovery_days: int = 8) -> Dict:
        """
        宽口径回测:
        - 只看创业板「急跌后8日大涨」
        - 突破60日高点
        - 显著跑赢上证
        """
        returns, close = self.parent.prepare_data()

        if '创业板指' not in close.columns:
            return {'error': '缺少创业板指数据'}

        cyb = close['创业板指']
        sh = close.get('上证指数', cyb)

        cyb_drop = (cyb.shift(recovery_days) / cyb.shift(recovery_days + 5) - 1)  # 5日涨幅, 结束于8天前
        cyb_recovery = cyb.pct_change(recovery_days)

        matches = []
        for i in range(recovery_days + 5, len(close) - 20):
            cyb_d = cyb_drop.iloc[i]
            cyb_r = cyb_recovery.iloc[i]

            if pd.isna(cyb_d):
                continue
            if cyb_d > drop_threshold:
                continue
            if cyb_r < recovery_threshold:
                continue

            # 突破60日高点
            cyb_60h = cyb.iloc[max(0, i - 59):i + 1].max()
            cyb_dist = (cyb.iloc[i] - cyb_60h) / cyb_60h
            if cyb_dist < -0.03:
                continue

            # 跑赢上证
            if i >= 7:
                cyb_8d = returns['创业板指'].iloc[i - 7:i + 1].sum()
                sh_8d = returns['上证指数'].iloc[i - 7:i + 1].sum()
                excess = cyb_8d - sh_8d
                if excess < 0.02:
                    continue
            else:
                excess = 0

            row = {'date': close.index[i], 'cyb_excess': excess}
            for n in [1, 2, 3, 5, 10, 20]:
                if i + n < len(close):
                    for name in ['创业板指', '科创50', '上证指数']:
                        if name in close.columns:
                            s = close[name]
                            fwd = (s.iloc[i + n] - s.iloc[i]) / s.iloc[i]
                            row[f'{name}_fwd{n}d'] = fwd
            matches.append(row)

        if not matches:
            return {'wide_matches': 0, 'message': '无匹配的宽口径场景'}

        df = pd.DataFrame(matches)
        result = self._summarize_backtest(df, 'wide')
        result.update(self._unconditional_baseline(cyb))
        return result

    @staticmethod
    def _unconditional_baseline(cyb: pd.Series) -> Dict:
        """创业板指无条件胜率基准（review §4.3：胜率调整的正确对照是无条件基准，
        而非 50%——全样本 5 日自然胜率约 51.5%）。"""
        out = {}
        for n in [5, 20]:
            fwd = cyb.pct_change(n).dropna()
            if len(fwd) > 0:
                out[f'创业板指_{n}d_uncond_win_rate'] = float((fwd > 0).mean())
        return out

    def _summarize_backtest(self, df: pd.DataFrame, label: str) -> Dict:
        """汇总回测统计"""
        result = {
            f'{label}_matches': len(df),
            f'{label}_dates': df['date'].dt.strftime('%Y-%m-%d').tolist()[:10],  # 最近10个
        }

        # 各指数各周期的统计
        for name in ['创业板指', '科创50', '上证指数']:
            for n in [1, 2, 3, 5, 10, 20]:
                col = f'{name}_fwd{n}d'
                if col in df.columns:
                    fwd = df[col].dropna()
                    if len(fwd) > 0:
                        result[f'{name}_{n}d_win_rate'] = (fwd > 0).mean()
                        result[f'{name}_{n}d_mean'] = fwd.mean()
                        result[f'{name}_{n}d_median'] = fwd.median()
                        result[f'{name}_{n}d_max'] = fwd.max()
                        result[f'{name}_{n}d_min'] = fwd.min()
                        result[f'{name}_{n}d_std'] = fwd.std()

        return result
