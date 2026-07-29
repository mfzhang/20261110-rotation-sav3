#!/usr/bin/env python3
# wave_analyzer.py
# 波浪结构分析

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("market_timing")


class WaveAnalyzer:
    """波浪结构分析"""

    def __init__(self, parent):
        self.parent = parent

    def analyze_wave_structure(self) -> Dict[str, Dict]:
        """
        艾略特波浪结构判断 (简化版):
        - 识别是否处于 B浪反弹 vs 新推动浪
        - 判断成长/科技指数级弱转强是否成立
        """
        _, close_df = self.parent.prepare_data()
        result = {}

        for name in close_df.columns:
            s = close_df[name].dropna()
            if len(s) < 120:
                continue

            wave = {}
            current = s.iloc[-1]

            # 找最近一个大级别的高低点
            # 120日最高点和最低点
            hh_120 = s.iloc[-120:].max()
            ll_120 = s.iloc[-120:].min()
            hh_idx = s.iloc[-120:].idxmax()
            ll_idx = s.iloc[-120:].idxmin()

            # 从最高点回撤幅度
            drawdown_from_high = (current - hh_120) / hh_120
            # 从最低点反弹幅度
            rally_from_low = (current - ll_120) / ll_120

            wave['120d_high'] = round(float(hh_120), 2)
            wave['120d_low'] = round(float(ll_120), 2)
            wave['drawdown_from_high'] = round(float(drawdown_from_high), 4)
            wave['rally_from_low'] = round(float(rally_from_low), 4)

            # 判断: B浪 vs 新推动浪
            # B浪特征: 从高点大幅回撤后反弹，但反弹不超过前高的61.8%-78.6%
            # 新推动浪: 反弹超过前高或接近前高，且有放量配合

            if drawdown_from_high < -0.10:
                # 出现过大回撤
                # 回撤修复比例 = (现价-区间低点)/(区间高点-区间低点)
                # 原公式 rally_from_low/abs(drawdown_from_high) 分母基准混用
                # （例：100→50→80 真实修复 60%，旧式算出 120% 误判新推动浪，review §4.3）
                retrace_ratio = (current - ll_120) / (hh_120 - ll_120) if hh_120 > ll_120 else 0.0
                wave['retrace_ratio'] = round(float(retrace_ratio), 4)

                if retrace_ratio < 0.382:
                    wave['wave_judgment'] = '弱势反弹 (可能只是下跌中继)'
                    wave['wave_confidence'] = '中等'
                elif retrace_ratio < 0.618:
                    wave['wave_judgment'] = 'B浪反弹特征 (关注是否突破61.8%)'
                    wave['wave_confidence'] = '较高'
                elif retrace_ratio < 0.786:
                    wave['wave_judgment'] = '强反弹，接近B浪极限 (若突破则确认新推动浪)'
                    wave['wave_confidence'] = '较高'
                else:
                    wave['wave_judgment'] = '已突破78.6%回撤，可能是新推动浪'
                    wave['wave_confidence'] = '较低 (需更多确认)'
            else:
                wave['wave_judgment'] = '无大级别回撤，趋势延续中'
                wave['wave_confidence'] = '低'
                wave['retrace_ratio'] = None

            # 成交量配合判断
            if name in self.parent.fetcher._raw_data and 'vol' in self.parent.fetcher._raw_data[name].columns:
                vol = self.parent.fetcher._raw_data[name]['vol']
                if len(vol) >= 20:
                    vol_5d = vol.iloc[-5:].mean()
                    vol_20d = vol.iloc[-20:].mean()
                    wave['volume_ratio'] = round(float(vol_5d / vol_20d), 2) if vol_20d > 0 else 1.0
                    wave['volume_judgment'] = '放量' if wave['volume_ratio'] > 1.2 else \
                        ('缩量' if wave['volume_ratio'] < 0.8 else '正常')

            result[name] = wave

        # 全局判断: 成长/科技指数级弱转强
        result['global'] = self._judge_growth_turnaround(result)

        return result

    def _judge_growth_turnaround(self, wave_result: Dict) -> Dict:
        """判断成长/科技指数是否弱转强"""
        growth = {}
        for name in ['创业板指', '科创50', '中证1000']:
            if name in wave_result:
                w = wave_result[name]
                growth[name] = {
                    'drawdown': w.get('drawdown_from_high', 0),
                    'rally': w.get('rally_from_low', 0),
                    'retrace': w.get('retrace_ratio'),
                    'judgment': w.get('wave_judgment', ''),
                }

        # 价值指数 (上证/沪深300)
        value = {}
        for name in ['上证指数', '沪深300']:
            if name in wave_result:
                w = wave_result[name]
                value[name] = w.get('drawdown_from_high', 0)

        # 判断
        growth_rallying = all(
            info['rally'] > 0.05
            for info in growth.values()
        ) if growth else False

        value_weak = all(
            v < -0.02
            for v in value.values()
        ) if value else False

        conclusion = {
            'growth_indices': growth,
            'value_indices': value,
            'growth_rallying': growth_rallying,
        }

        if growth_rallying and value_weak:
            conclusion['turnaround_judgment'] = '成长/科技指数级弱转强局部成立，全市场级尚未完全成立'
            conclusion['turnaround_confidence'] = '中等'
        elif growth_rallying:
            conclusion['turnaround_judgment'] = '成长/科技反弹中，弱转强待确认'
            conclusion['turnaround_confidence'] = '低'
        else:
            conclusion['turnaround_judgment'] = '成长/科技尚未出现弱转强信号'
            conclusion['turnaround_confidence'] = '低'

        return conclusion
