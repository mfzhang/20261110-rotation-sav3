#!/usr/bin/env python3
# /tmp/sav3/market_timing.py
# 大盘择时模块 — 历史类比回测 + 多剧本概率预测 + 关键点位分析
#
# 基于原型模型设计:
#   - 双口径回测: 严格口径(创业板+科创50) / 宽口径(仅创业板)
#   - 三剧本概率框架: 强势分歧后逼空 / 直接强逼空 / 高开兑现失败
#   - 关键点位: 支撑/压力/均线/布林带/斐波那契
#   - VWAP替代: 日线级别确认/失效条件
#
# 用法:
#   from market_timing import MarketTiming
#   mt = MarketTiming()
#   result = mt.run_full_analysis()
#   print(result['summary'])

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_fetcher import _get_pro, fetch_indices_daily

from index_data import IndexDataFetcher, INDEX_CODES
from scene_matcher import SceneMatcher, SIMILARITY_INDEX_WEIGHTS
from pattern_backtester import PatternBacktester
from key_levels import KeyLevelsCalculator
from wave_analyzer import WaveAnalyzer
from scenario_generator import ScenarioGenerator

logger = logging.getLogger("market_timing")


class MarketTiming:
    """大盘择时分析引擎"""

    def __init__(self, end_date: str = None, lookback_years: int = 6,
                 start_date: str = '2015-01-01'):
        """
        Args:
            end_date: 分析目标日期 (YYYY-MM-DD)，默认最新交易日
            lookback_years: 回看年数
            start_date: 最早数据日期
        """
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.lookback_years = lookback_years
        self.start_date = start_date

        # 懒加载数据容器
        self._raw_data: Dict[str, pd.DataFrame] = {}
        self._daily_returns: pd.DataFrame = None   # date x index_name
        self._daily_close: pd.DataFrame = None

        # 各职责组件
        self.fetcher = IndexDataFetcher(start_date=self.start_date, end_date=self.end_date)
        self.scene_matcher = SceneMatcher()
        self.backtester = PatternBacktester(self)
        self.key_levels = KeyLevelsCalculator(self)
        self.wave_analyzer = WaveAnalyzer(self)
        self.scenario_generator = ScenarioGenerator(self)

    # ═══════════════════════════════════════════════
    # 数据获取 (委托给 IndexDataFetcher)
    # ═══════════════════════════════════════════════

    def fetch_all_indices(self, refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """拉取全部六大指数日K数据，统一走 data_fetcher。"""
        self._raw_data = self.fetcher.fetch_all_indices(refresh=refresh)
        return self._raw_data

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """准备统一数据表: daily_returns, daily_close (date x index_name)"""
        self._daily_returns, self._daily_close = self.fetcher.prepare_data()
        self._raw_data = self.fetcher._raw_data

        # 为 SceneMatcher 注入数据
        self.scene_matcher._daily_returns = self._daily_returns
        self.scene_matcher._daily_close = self._daily_close

        return self._daily_returns, self._daily_close

    # ═══════════════════════════════════════════════
    # 核心指标计算 (委托给 KeyLevelsCalculator)
    # ═══════════════════════════════════════════════

    def compute_bollinger(self, close: pd.Series, window: int = 20, n_std: float = 2.0
                          ) -> Dict[str, pd.Series]:
        """布林带"""
        return self.key_levels.compute_bollinger(close, window, n_std)

    # ═══════════════════════════════════════════════
    # 历史类比回测 (委托给 SceneMatcher / PatternBacktester)
    # ═══════════════════════════════════════════════

    def compute_state_vector(self, date: pd.Timestamp) -> Dict[str, float]:
        """计算某日的状态向量，用于相似度匹配"""
        self.prepare_data()
        return self.scene_matcher.compute_state_vector(date)

    def find_similar_scenes(self, current_date: pd.Timestamp = None,
                            top_n: int = 20, min_distance: int = 20) -> pd.DataFrame:
        """
        历史类比搜索: 找出历史上与当前状态最相似的场景。

        Returns:
            DataFrame with columns: date, similarity, and subsequent N-day returns
        """
        self.prepare_data()
        return self.scene_matcher.find_similar_scenes(current_date, top_n, min_distance)

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
        return self.backtester.backtest_strict_criteria(drop_threshold,
                                                         recovery_threshold,
                                                         recovery_days)

    def backtest_wide_criteria(self, drop_threshold: float = -0.05,
                                recovery_threshold: float = 0.05,
                                recovery_days: int = 8) -> Dict:
        """
        宽口径回测:
        - 只看创业板「急跌后8日大涨」
        - 突破60日高点
        - 显著跑赢上证
        """
        return self.backtester.backtest_wide_criteria(drop_threshold,
                                                       recovery_threshold,
                                                       recovery_days)

    # ═══════════════════════════════════════════════
    # 关键点位计算 (委托给 KeyLevelsCalculator)
    # ═══════════════════════════════════════════════

    def compute_key_levels(self) -> Dict[str, Dict]:
        """计算各指数的关键支撑/压力位"""
        return self.key_levels.compute_key_levels()

    # ═══════════════════════════════════════════════
    # 波浪结构分析 (委托给 WaveAnalyzer)
    # ═══════════════════════════════════════════════

    def analyze_wave_structure(self) -> Dict[str, Dict]:
        """
        艾略特波浪结构判断 (简化版):
        - 识别是否处于 B浪反弹 vs 新推动浪
        - 判断成长/科技指数级弱转强是否成立
        """
        return self.wave_analyzer.analyze_wave_structure()

    # ═══════════════════════════════════════════════
    # 多剧本概率预测 (委托给 ScenarioGenerator)
    # ═══════════════════════════════════════════════

    def generate_scenarios(self) -> Dict:
        """
        生成三剧本概率预测:
        剧本一: 强势分歧后继续逼空 (50-55%)
        剧本二: 直接强逼空 (20-25%)
        剧本三: 高开兑现，逼空失败 (20-25%)
        """
        return self.scenario_generator.generate_scenarios()

    # ═══════════════════════════════════════════════
    # 日线确认/失效条件 (委托给 ScenarioGenerator)
    # ═══════════════════════════════════════════════

    def get_daily_confirmation_rules(self) -> Dict:
        """
        VWAP替代方案 — 日线级别的确认/失效条件:
        由于无法获取日内数据，使用日线指标替代VWAP判断。
        """
        return self.scenario_generator.get_daily_confirmation_rules()

    # ═══════════════════════════════════════════════
    # 综合输出
    # ═══════════════════════════════════════════════

    def run_full_analysis(self) -> Dict:
        """
        运行完整大盘择时分析，返回结构化结果。

        Returns:
            dict with keys:
            - summary: 一段文字总结
            - scenarios: 三剧本预测
            - key_levels: 各指数关键点位
            - wave_structure: 波浪结构分析
            - backtest: 历史回测统计
            - confirmation_rules: 日线确认规则
            - similar_scenes: 历史相似场景
        """
        logger.info("=" * 50)
        logger.info(f"大盘择时分析 — {self.end_date}")
        logger.info("=" * 50)

        result = {}

        # 1. 历史类比回测
        logger.info("[1/5] 历史类比回测...")
        strict_bt = self.backtest_strict_criteria()
        wide_bt = self.backtest_wide_criteria()
        similar = self.find_similar_scenes(top_n=15)
        result['backtest'] = {
            'strict': strict_bt,
            'wide': wide_bt,
        }
        result['similar_scenes'] = self._format_similar(similar)

        # 2. 关键点位
        logger.info("[2/5] 关键点位计算...")
        result['key_levels'] = self.compute_key_levels()

        # 3. 波浪结构
        logger.info("[3/5] 波浪结构分析...")
        result['wave_structure'] = self.analyze_wave_structure()

        # 4. 多剧本概率
        logger.info("[4/5] 多剧本概率预测...")
        result['scenarios'] = self.generate_scenarios()

        # 5. 确认规则
        logger.info("[5/5] 确认规则...")
        result['confirmation_rules'] = self.get_daily_confirmation_rules()

        # 6. 数据健康（供日报系统健康区：PSM 六指数完整性是 P0-3 的防线）
        try:
            raw = self.fetcher._raw_data or self.fetch_all_indices()
            dh = {}
            for name in INDEX_CODES:
                df = raw.get(name)
                if df is None or df.empty:
                    dh[name] = {'rows': 0, 'latest': None}
                else:
                    dh[name] = {'rows': len(df),
                                'latest': df.index.max().strftime('%Y-%m-%d')}
            result['data_health'] = dh
            result['data_missing'] = [n for n, v in dh.items() if v['rows'] == 0]
            if result['data_missing']:
                logger.error(f"PSM 指数数据缺失: {result['data_missing']}")
        except Exception as e:
            logger.warning(f"PSM 数据健康采集失败: {e}")

        # 生成摘要
        result['summary'] = self.scenario_generator._generate_summary(result)

        logger.info("分析完成 ✅")
        return result

    def _format_similar(self, df: pd.DataFrame) -> List[Dict]:
        """格式化相似场景输出"""
        if df is None or df.empty:
            return []

        df = df.sort_values('similarity', ascending=False).head(10)
        scenes = []
        for _, row in df.iterrows():
            scene = {
                'date': row['date'].strftime('%Y-%m-%d'),
                'similarity': round(float(row['similarity']), 3),
            }
            for col in row.index:
                if 'fwd' in col:
                    if not pd.isna(row[col]):
                        scene[col] = round(float(row[col]), 4)
            scenes.append(scene)
        return scenes


# ═══════════════════════════════════════════════
# 自检入口
# ═══════════════════════════════════════════════

def run_self_check():
    """独立可测: python3 -m market_timing"""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    print("=" * 60)
    print("market_timing 自检")
    print("=" * 60)

    mt = MarketTiming()

    # 1. 数据拉取
    print("\n>>> 1. 指数数据拉取")
    data = mt.fetch_all_indices()
    for name, df in data.items():
        print(f"  {name}: {len(df)} 行, {df.index[0].date()} ~ {df.index[-1].date()}")

    returns, close = mt.prepare_data()
    print(f"  统一数据: {len(close)} 交易日, {list(close.columns)}")

    # 2. 关键点位
    print("\n>>> 2. 关键点位")
    levels = mt.compute_key_levels()
    for name in ['创业板指', '科创50', '上证指数', '沪深300', '中证1000']:
        if name in levels:
            lv = levels[name]
            cur = lv.get('current_close', 'N/A')
            ma20 = lv.get('MA20', 'N/A')
            ma60 = lv.get('MA60', 'N/A')
            res = lv.get('key_resistance', [])[:2]
            sup = lv.get('key_support', [])[:2]
            fib_618 = lv.get('fib_618', 'N/A')
            print(f"  {name}: 现价{cur}, MA20={ma20}, MA60={ma60}")
            print(f"    压力: {res}")
            print(f"    支撑: {sup}")
            print(f"    斐波那契0.618: {fib_618}")

    # 3. 波浪结构
    print("\n>>> 3. 波浪结构")
    wave = mt.analyze_wave_structure()
    for name in ['创业板指', '科创50', '上证指数']:
        if name in wave:
            w = wave[name]
            print(f"  {name}: {w.get('wave_judgment','N/A')}")
            if w.get('retrace_ratio'):
                print(f"    回撤比率: {w['retrace_ratio']:.1%}")
            if w.get('volume_ratio'):
                print(f"    量比(5/20): {w['volume_ratio']:.1f}x ({w.get('volume_judgment','')})")
    if 'global' in wave:
        print(f"  全局: {wave['global'].get('turnaround_judgment','')}")

    # 4. 历史回测
    print("\n>>> 4. 历史类比回测")
    wide = mt.backtest_wide_criteria()
    strict = mt.backtest_strict_criteria()
    print(f"  严格口径: {strict.get('strict_matches', 0)} 次匹配")
    print(f"  宽口径: {wide.get('wide_matches', 0)} 次匹配")

    for bt, label in [(wide, '宽口径'), (strict, '严格口径')]:
        for n in [1, 2, 3, 5, 10]:
            wr = bt.get(f'创业板指_{n}d_win_rate')
            mean_r = bt.get(f'创业板指_{n}d_mean')
            if wr is not None:
                print(f"  {label} 创业板 {n}日: 胜率{wr:.1%} 均值{mean_r:+.2%}")

    # 5. 多剧本
    print("\n>>> 5. 三剧本（仪表盘模式：n<20 不输出概率）")
    scenarios = mt.generate_scenarios()
    for key in ['scenario_1', 'scenario_2', 'scenario_3']:
        s = scenarios.get(key, {})
        prob = s.get('probability')
        prob_str = f"{prob:.1%}" if prob is not None else "—"
        print(f"  {s.get('name','')}: {prob_str} ({s.get('probability_range','')})")
    bench = scenarios.get('current_benchmark', {})
    print(f"  成长站上MA60: {bench.get('growth_above_ma60')}")
    print(f"  成长强修复: {bench.get('growth_strong_recovery')}")
    dash = scenarios.get('dashboard', {})
    print(f"  加权20日涨跌幅: {dash.get('weighted_ret_20d')}")
    print(f"  样本量: {dash.get('sample')}")

    # 6. 确认规则
    print("\n>>> 6. 日线确认状态")
    rules = mt.get_daily_confirmation_rules()
    for name, status in rules.get('current_status', {}).items():
        yang = '阳' if status.get('yang_xian') else '阴'
        above = '站上' if status.get('above_ma5') else '跌破'
        print(f"  {name}: {yang}线, {above}MA5")

    # 7. 综合输出
    print("\n>>> 7. 综合摘要")
    full = mt.run_full_analysis()
    print(full['summary'])

    print("\n" + "=" * 60)
    print("自检完成 ✅")
    return full


if __name__ == '__main__':
    run_self_check()
