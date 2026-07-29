#!/usr/bin/env python3
# scenario_generator.py
# 多剧本概率预测与确认规则

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from index_data import INDEX_CODES
from scene_matcher import SIMILARITY_INDEX_WEIGHTS

logger = logging.getLogger("market_timing")

# 概率输出的最小样本量（宽口径历史匹配次数）
# 实证：宽口径匹配 ~10 次，n<20 时精确到个位的概率是"小样本精确感"假象
PSM_MIN_SAMPLE = 20


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple:
    """二项胜率的 Wilson 置信区间（小样本比正态近似稳健）。"""
    if p is None or n <= 0:
        return (None, None)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _mean_ci(mean, std, n, z: float = 1.96) -> tuple:
    """均值的渐近置信区间 mean ± z·σ/√n。"""
    if mean is None or std is None or n <= 1 or pd.isna(std):
        return (None, None)
    half = z * std / np.sqrt(n)
    return (mean - half, mean + half)


class ScenarioGenerator:
    """多剧本概率预测与确认规则（2026-07-19 起：指标仪表盘模式，见 _build_dashboard）"""

    def __init__(self, parent):
        self.parent = parent

    def _build_dashboard(self, returns, close, wide_bt: Dict, strict_bt: Dict) -> Dict:
        """PSM 指标仪表盘（替代精确概率的主输出）。

        输出六项可复核指标：
          1. MA60 状态（各指数站上/跌破 + 连续天数）
          2. 距 60 日高点距离
          3. 波动率状态（20 日波动率 + 近 252 日分位 → 高/中/低）
          4. 加权 20 日涨跌幅（SIMILARITY_INDEX_WEIGHTS 成长加权）
          5. 样本量（宽/严格口径历史匹配次数）
          6. 置信区间（仅 n≥20：胜率 Wilson 95% CI + 均值渐近 CI）

        设计动机：宽口径历史匹配实证仅 ~10 次，精确概率是"小样本精确感"
        假象；仪表盘把判断依据全部摊开，样本不足时明确不输出概率。
        """
        dash = {'indices': {}}
        weighted_num, weighted_den = 0.0, 0.0

        for name in close.columns:
            s = close[name].dropna()
            if len(s) < 60:
                continue
            cur = float(s.iloc[-1])
            ma60_series = s.rolling(60).mean()
            ma60 = float(ma60_series.iloc[-1])
            hh60 = float(s.iloc[-60:].max())

            # 连续站上 MA60 天数
            above = s > ma60_series
            cnt = 0
            for v in reversed(above.values):
                if not pd.isna(v) and v:
                    cnt += 1
                else:
                    break

            # 波动率状态：20 日收益 std 在近 252 日的分位
            r = returns[name].dropna()
            vol20_series = r.rolling(20).std().dropna()
            vol20 = float(vol20_series.iloc[-1]) if len(vol20_series) else float('nan')
            vol_hist = vol20_series.iloc[-252:]
            vol_pct = float((vol_hist <= vol20).mean()) if len(vol_hist) >= 20 else float('nan')
            if pd.isna(vol_pct):
                vol_state = '未知'
            elif vol_pct >= 0.8:
                vol_state = '高'
            elif vol_pct <= 0.2:
                vol_state = '低'
            else:
                vol_state = '中'

            ret20 = float(cur / s.iloc[-21] - 1) if len(s) >= 21 else float('nan')

            dash['indices'][name] = {
                'close': round(cur, 2),
                'above_ma60': cur > ma60,
                'ma60_days': cnt,
                'dist_60high': round((cur - hh60) / hh60, 4),
                'vol20': round(vol20, 4) if not pd.isna(vol20) else None,
                'vol_pct': round(vol_pct, 2) if not pd.isna(vol_pct) else None,
                'vol_state': vol_state,
                'ret_20d': round(ret20, 4) if not pd.isna(ret20) else None,
            }

            w = SIMILARITY_INDEX_WEIGHTS.get(name)
            if w and not pd.isna(ret20):
                weighted_num += w * ret20
                weighted_den += w

        dash['weighted_ret_20d'] = (round(weighted_num / weighted_den, 4)
                                    if weighted_den > 0 else None)

        n_wide = int(wide_bt.get('wide_matches', 0) or 0)
        n_strict = int(strict_bt.get('strict_matches', 0) or 0)
        dash['sample'] = {
            'n_wide': n_wide,
            'n_strict': n_strict,
            'sufficient': n_wide >= PSM_MIN_SAMPLE,
        }

        # 胜率/均值 + 置信区间（仅样本充足时输出）
        stats = None
        if n_wide >= PSM_MIN_SAMPLE:
            p5 = wide_bt.get('创业板指_5d_win_rate')
            p20 = wide_bt.get('创业板指_20d_win_rate')
            stats = {
                'cyb_5d_win_rate': p5,
                'cyb_5d_ci': _wilson_ci(p5, n_wide),
                'cyb_20d_win_rate': p20,
                'cyb_20d_ci': _wilson_ci(p20, n_wide),
                'cyb_5d_mean': wide_bt.get('创业板指_5d_mean'),
                'cyb_5d_mean_ci': _mean_ci(wide_bt.get('创业板指_5d_mean'),
                                           wide_bt.get('创业板指_5d_std'), n_wide),
                'cyb_5d_uncond': wide_bt.get('创业板指_5d_uncond_win_rate'),
            }
        dash['stats'] = stats
        return dash

    def _dynamic_target_zones(self):
        """用现算关键点位生成三剧本目标位（替代硬编码——review §4.3：
        硬编码目标价曾偏离现价 17-20%，每天原样印出）。

        返回 (zones_up, zones_break, zones_down)；失败返回 None 由调用方降级。
        """
        try:
            levels = self.parent.compute_key_levels()
        except Exception as e:
            logger.warning(f"动态目标位生成失败，降级为均线描述: {e}")
            return None

        zones_up, zones_break, zones_down = {}, {}, {}
        for name in ['创业板指', '科创50', '上证指数']:
            lv = levels.get(name)
            if not lv or lv.get('current_close') is None:
                continue
            res = lv.get('key_resistance', [])
            sup = lv.get('key_support', [])
            r1 = res[0] if len(res) > 0 else None
            r2 = res[1] if len(res) > 1 else None
            s1 = sup[0] if len(sup) > 0 else None
            s2 = sup[1] if len(sup) > 1 else None
            if r1 and r2:
                zones_up[name] = f'{r1:.0f} → {r2:.0f}'
                zones_break[name] = f'突破{r1:.0f}看{r2:.0f}'
            elif r1:
                zones_up[name] = f'{r1:.0f}'
                zones_break[name] = f'站稳{r1:.0f}上方'
            if s1 and s2:
                zones_down[name] = f'回踩{s1:.0f}/{s2:.0f}支撑'
            elif s1:
                zones_down[name] = f'回踩{s1:.0f}支撑'
        return zones_up, zones_break, zones_down

    def generate_scenarios(self) -> Dict:
        """
        生成三剧本 + 指标仪表盘（2026-07-19 起为仪表盘模式）:
        - 主输出为 dashboard 六项指标（MA60/距60日高点/波动率/加权20日/样本量/CI）
        - 三剧本保留定性结构（触发条件/失效信号/目标位）
        - 精确概率仅在宽口径样本 n≥20 时输出（scenarios['probabilities_available']）
        """
        # 获取当前状态
        returns, close = self.parent.prepare_data()
        current_date = close.index[-1]

        # 计算当前各指数关键指标
        current = {}
        for name in close.columns:
            s = close[name]
            if len(s) < 60:
                continue
            cur = float(s.iloc[-1])
            ma20 = float(s.rolling(20).mean().iloc[-1])
            ma60 = float(s.rolling(60).mean().iloc[-1])
            hh_20 = float(s.iloc[-20:].max())
            ll_20 = float(s.iloc[-20:].min())

            current[name] = {
                'close': cur,
                'MA20': round(ma20, 2),
                'MA60': round(ma60, 2),
                'above_ma20': cur > ma20,
                'above_ma60': cur > ma60,
                'dist_20high': round((cur - hh_20) / hh_20, 4),
                'dist_20low': round((cur - ll_20) / ll_20, 4),
                'ret_5d': round(float(returns[name].iloc[-5:].sum() if len(returns) >= 5 else 0), 4),
                'ret_8d': round(float(returns[name].iloc[-8:].sum() if len(returns) >= 8 else 0), 4),
            }

        # 创业板相对上证超额
        if '创业板指' in returns.columns and '上证指数' in returns.columns:
            excess_8d = returns['创业板指'].iloc[-8:].sum() - returns['上证指数'].iloc[-8:].sum()
            current['excess'] = {'创业板vs上证_8d': round(float(excess_8d), 4)}

        # ── 场景判断逻辑 ──
        cyb = current.get('创业板指', {})
        kc50 = current.get('科创50', {})
        excess = current.get('excess', {}).get('创业板vs上证_8d', 0)

        # 条件判断
        growth_above_ma60 = cyb.get('above_ma60', False) and kc50.get('above_ma60', False)
        growth_strong_recovery = (cyb.get('ret_8d', 0) > 0.05 and kc50.get('ret_8d', 0) > 0.05)
        growth_near_high = (abs(cyb.get('dist_20high', -1)) < 0.03 and
                           abs(kc50.get('dist_20high', -1)) < 0.03)
        excess_strong = excess > 0.03

        # ── MA60 站上持续时间 ──
        ma60_consecutive_days = None
        if growth_above_ma60:
            conseq_days_list = []
            for name in ['创业板指', '科创50']:
                if name in close.columns:
                    s = close[name]
                    ma60_series = s.rolling(60).mean()
                    above_series = s > ma60_series
                    cnt = 0
                    for j in range(len(above_series) - 1, -1, -1):
                        val = above_series.iloc[j]
                        if not pd.isna(val) and bool(val):
                            cnt += 1
                        else:
                            break
                    conseq_days_list.append(cnt)
            if conseq_days_list:
                ma60_consecutive_days = min(conseq_days_list)

        # ── 从历史回测获取概率基准 ──
        wide_bt = self.parent.backtest_wide_criteria()
        strict_bt = self.parent.backtest_strict_criteria()

        # 创业板后续1-5日胜率
        cyb_win_rate = wide_bt.get('创业板指_5d_win_rate',
                       strict_bt.get('创业板指_5d_win_rate', 0.5))
        n_wide = wide_bt.get('wide_matches', 0)
        uncond_win = wide_bt.get('创业板指_5d_uncond_win_rate')

        # ── 动态目标位（现算关键点位，替代硬编码）──
        dyn_zones = self._dynamic_target_zones()
        if dyn_zones:
            zones_up, zones_break, zones_down = dyn_zones
        else:
            zones_up = zones_break = zones_down = {}

        def _zone(zones, name, fallback):
            return zones.get(name, fallback)

        # ── 三剧本 ──
        scenarios = {
            'scenario_1': {
                'name': '强势分歧后继续逼空',
                'probability': 0.50 if growth_strong_recovery and growth_near_high else 0.35,
                'probability_range': '50-55%' if growth_strong_recovery else '35-45%',
                'trigger_conditions': [
                    '涨停/大涨后次日分歧低开',
                    '9:35后核心票站稳日线MA5/MA10',
                    '成交量不萎缩超20%',
                    '创业板不破前日低点',
                ],
                'path_description': (
                    '周初/次日出现分歧调整(低开或盘中急跌)，'
                    '但核心成长票在关键均线处获得支撑，'
                    '午后或次日重新走强，开启第二波逼空。'
                ),
                'invalidation_signals': [
                    '创业板跌破前日低点且30分钟内未收回',
                    '科创50跌幅超2%',
                    '核心票(光模块/AI芯片)大面积破MA10',
                ],
                'target_zones': {
                    '创业板指': _zone(zones_up, '创业板指', '见上方关键点位'),
                    '科创50': _zone(zones_up, '科创50', '见上方关键点位'),
                    '上证指数': _zone(zones_up, '上证指数', '见上方关键点位'),
                },
            },
            'scenario_2': {
                'name': '直接强逼空',
                'probability': 0.22 if growth_near_high else 0.15,
                'probability_range': '20-25%' if growth_near_high else '15-20%',
                'trigger_conditions': [
                    '次日直接高开且不回补缺口',
                    '9:35时成长指数涨幅>1%',
                    '核心票集体高开高走',
                    '成交量较前日放大20%以上',
                ],
                'path_description': (
                    '延续强势，次日直接高开高走，'
                    '成长指数单日涨幅2-3%，'
                    '强势突破近期高点，开启主升浪。'
                ),
                'invalidation_signals': [
                    '高开后30分钟内回补缺口',
                    '核心票冲高回落留长上影',
                    '成交量未能持续放大',
                ],
                'target_zones': {
                    '创业板指': _zone(zones_break, '创业板指', '看上方次級压力'),
                    '科创50': _zone(zones_break, '科创50', '看上方次級压力'),
                    '上证指数': _zone(zones_break, '上证指数', '看上方次級压力'),
                },
            },
            'scenario_3': {
                'name': '高开兑现，逼空失败',
                'probability': 0.23 if growth_strong_recovery else 0.30,
                'probability_range': '20-25%' if growth_strong_recovery else '25-35%',
                'trigger_conditions': [
                    '高开后快速回落',
                    '核心票大面积翻绿',
                    '成交量放大但价格不涨(放量滞涨)',
                    '创业板跌破开盘价',
                ],
                'path_description': (
                    '借助前日强势高开，但获利盘涌出，'
                    '高开后持续走低，日线收阴或长上影，'
                    '逼空失败，可能回踩MA10/MA20支撑。'
                ),
                'invalidation_signals': [
                    '创业板高开后翻绿',
                    '核心票跌破VWAP(日线替代: 跌破开盘价)',
                    '科创50由涨转跌',
                    '跌停家数开始增加',
                ],
                'target_zones': {
                    '创业板指': _zone(zones_down, '创业板指', '回踩MA10/MA20支撑'),
                    '科创50': _zone(zones_down, '科创50', '回踩MA10/MA20支撑'),
                    '上证指数': _zone(zones_down, '上证指数', '可能测试近期低点'),
                },
            },
        }

        # ── PSM 指标仪表盘（主输出，替代精确概率）──
        dashboard = self._build_dashboard(returns, close, wide_bt, strict_bt)

        # ── 概率输出门禁：n<20 不输出概率 ──
        # （review §4.3 + 2026-07-19 模型层修复：宽口径实证 ~10 次匹配，
        # 精确到个位的剧本概率是小样本精确感假象；n≥20 才保留概率框架）
        probabilities_available = n_wide >= PSM_MIN_SAMPLE

        # ── 根据历史回测调整概率 ──
        # 门槛（review §4.3）：样本 n≥20 才调整；基准是无条件胜率而非 50%；
        # 该形态实证统计偏空（宽口径 5 日胜率 40% vs 无条件 51.5%），
        # n<20 时不输出基于历史的概率偏移，避免小样本精确感。
        ma60_note = None
        adj = 0.0
        if probabilities_available and cyb_win_rate and uncond_win:
            edge = cyb_win_rate - uncond_win
            adj = float(np.clip(edge * 0.5, -0.10, 0.10))

            # MA60 站上持续时间降权: 刚站上<5天，调整量打折50%
            if growth_above_ma60 and ma60_consecutive_days is not None:
                if ma60_consecutive_days < 5:
                    adj *= 0.5

            scenarios['scenario_1']['probability'] = min(0.60,
                scenarios['scenario_1']['probability'] + adj)
            scenarios['scenario_2']['probability'] = min(0.30,
                scenarios['scenario_2']['probability'] + adj * 0.5)
            scenarios['scenario_3']['probability'] = max(0.15,
                scenarios['scenario_3']['probability'] - adj * 1.5)

        # MA60 长期站上提示：放入 current_benchmark（原注入 scenarios dict
        # 会在下方 sum(s['probability']...) 处 TypeError 崩掉整个 PSM 板块）
        if growth_above_ma60 and ma60_consecutive_days is not None \
                and ma60_consecutive_days > 60:
            ma60_note = '⚠️ 长期站上MA60(>60天)，注意overbought风险'

        if probabilities_available:
            # 归一化概率
            total_p = sum(s['probability'] for s in scenarios.values())
            if total_p > 0:
                for s in scenarios.values():
                    s['probability'] = round(s['probability'] / total_p, 3)
        else:
            # 样本不足：不输出精确概率，只保留剧本的定性结构（触发/失效条件）
            for s in scenarios.values():
                s['probability'] = None
                s['probability_range'] = f'样本不足(n={n_wide}<{PSM_MIN_SAMPLE})，不输出概率'
            logger.info(f"PSM 样本 n={n_wide}<{PSM_MIN_SAMPLE}，"
                        "概率输出抑制（仪表盘模式）")

        scenarios['probabilities_available'] = probabilities_available
        scenarios['dashboard'] = dashboard

        # ── 当前基准数据 ──
        scenarios['current_benchmark'] = {
            'date': current_date.strftime('%Y-%m-%d'),
            'weekday': ['一','二','三','四','五','六','日'][current_date.weekday()],
            'indices': current,
            'growth_above_ma60': growth_above_ma60,
            'growth_strong_recovery': growth_strong_recovery,
            'growth_near_high': growth_near_high,
            'excess_strong': excess_strong,
            'ma60_consecutive_days': ma60_consecutive_days,
            'ma60_note': ma60_note,
        }

        # ── 历史回测引用 ──
        scenarios['backtest_reference'] = {
            'strict_matches': strict_bt.get('strict_matches', 0),
            'wide_matches': n_wide,
            'cyb_5d_win_rate': wide_bt.get('创业板指_5d_win_rate',
                              strict_bt.get('创业板指_5d_win_rate')),
            'cyb_5d_mean': wide_bt.get('创业板指_5d_mean',
                          strict_bt.get('创业板指_5d_mean')),
            'cyb_5d_uncond_win_rate': uncond_win,
            'prob_adjusted': n_wide >= 20,  # 样本不足时概率未做历史调整
        }

        return scenarios

    def get_daily_confirmation_rules(self) -> Dict:
        """
        VWAP替代方案 — 日线级别的确认/失效条件:
        由于无法获取日内数据，使用日线指标替代VWAP判断。
        """
        _, close = self.parent.prepare_data()

        rules = {
            'description': '日线替代VWAP的确认/失效规则 (无日内数据)',
            'confirmation_conditions': {
                'trend': [
                    '收盘价 > 开盘价 (阳线)',
                    '收盘价站稳MA5上方',
                    '成交量 > 前5日均量的1.2倍 (放量确认)',
                    '下影线长度 < 实体长度 (无显著抛压)',
                ],
                'strength': [
                    '创业板指涨幅 > 上证指数涨幅 + 0.5%',
                    '科创50涨幅 > 0 (不拖后腿)',
                    '上涨家数 > 下跌家数 (广度确认)',
                ],
            },
            'invalidation_conditions': {
                'trend': [
                    '收盘价 < 开盘价 且 跌幅 > 1%',
                    '收盘价跌破MA5',
                    '上影线长度 > 实体长度 × 2 (长上影=抛压)',
                    '成交量放大但收阴 (放量滞涨/出货)',
                ],
                'breadth': [
                    '创业板指滞涨或转跌',
                    '科创50由涨转跌',
                    '核心观察标的多数翻绿',
                ],
            },
            'core_watchlist': [
                '中际旭创', '新易盛', '天孚通信', '寒武纪',
                '光迅科技', '工业富联', '东山精密', '胜宏科技',
                '沪电股份', '生益科技',
            ],
            'execution_rules': [
                '周初高开不追，等9:35后日线确认 (日线级别: 等收盘确认或次日确认)',
                '周初分歧看低吸机会 (日线: 回踩MA10/MA20)',
                '核心跌破MA10直接降级为观望',
                '不按预测交易，按确认交易',
            ],
        }

        # 当前各指数的日线确认状态
        status = {}
        for name in close.columns:
            s = close[name]
            if len(s) < 10:
                continue

            cur = s.iloc[-1]
            prev = s.iloc[-2]
            ma5 = s.rolling(5).mean().iloc[-1]

            # 检查是否有open数据
            has_open = name in self.parent.fetcher._raw_data and 'open' in self.parent.fetcher._raw_data[name].columns
            if has_open:
                raw = self.parent.fetcher._raw_data[name]
                open_today = raw['open'].iloc[-1] if 'open' in raw.columns else cur
                high_today = raw['high'].iloc[-1] if 'high' in raw.columns else cur
                low_today = raw['low'].iloc[-1] if 'low' in raw.columns else cur

                yang = cur >= open_today
                upper_shadow = high_today - max(open_today, cur)
                lower_shadow = min(open_today, cur) - low_today
                body = abs(cur - open_today)

                status[name] = {
                    'yang_xian': yang,
                    'above_ma5': cur > ma5,
                    'upper_shadow_ratio': round(float(upper_shadow / body), 2) if body > 0 else None,
                    'lower_shadow_ratio': round(float(lower_shadow / body), 2) if body > 0 else None,
                }
            else:
                status[name] = {
                    'above_ma5': cur > ma5,
                    'pct_chg': round(float((cur - prev) / prev * 100), 2),
                }

        rules['current_status'] = status
        return rules

    def _generate_summary(self, result: Dict) -> str:
        """生成文字摘要"""
        scenarios = result.get('scenarios', {})
        wave = result.get('wave_structure', {}).get('global', {})
        bt = result.get('backtest', {})

        lines = []
        lines.append(f"📊 大盘择时分析 — {self.parent.end_date}")
        lines.append("")

        # 当前状态
        bench = scenarios.get('current_benchmark', {})
        lines.append("【当前状态】")
        # 跳过分隔键如 'excess', 'growth_*'
        index_keys = [k for k in bench.get('indices', {}).keys()
                      if k in INDEX_CODES]
        for name in index_keys:
            info = bench['indices'][name]
            close_v = info.get('close', float('nan'))
            ret_5d = info.get('ret_5d', float('nan'))
            ret_8d = info.get('ret_8d', float('nan'))
            dist_20h = info.get('dist_20high', float('nan'))

            def _fmt(v, style='.1f'):
                if pd.isna(v) or v is None:
                    return 'N/A'
                if style == '.2%':
                    return f'{v:+.2%}'
                return f'{v:{style}}'

            lines.append(f"  {name}: {_fmt(close_v)} "
                        f"(5日:{_fmt(ret_5d,'.2%')}, 8日:{_fmt(ret_8d,'.2%')}, "
                        f"距20日高:{_fmt(dist_20h,'.2%')})")
        # 成长指数站上MA60状态 (含持续时间)
        ma60_days = bench.get('ma60_consecutive_days')
        if bench.get('growth_above_ma60'):
            if ma60_days is not None:
                if ma60_days < 5:
                    ma60_status = f'✅ (⚠️ 刚站上 {ma60_days}天)'
                elif ma60_days > 60:
                    ma60_status = f'✅ (⚠️ 长期站上 {ma60_days}天, overbought)'
                else:
                    ma60_status = f'✅ (站稳 {ma60_days}天)'
            else:
                ma60_status = '✅'
        else:
            ma60_status = '❌'
        lines.append(f"  成长指数站上MA60: {ma60_status}")
        if bench.get('ma60_note'):
            lines.append(f"  {bench['ma60_note']}")
        lines.append(f"  成长强修复信号: {'✅' if bench.get('growth_strong_recovery') else '❌'}")
        lines.append("")

        # ── PSM 指标仪表盘（主输出）──
        dash = scenarios.get('dashboard', {})
        lines.append("【PSM 指标仪表盘】")
        for name, info in dash.get('indices', {}).items():
            ma_str = '✅站上' if info.get('above_ma60') else '❌跌破'
            vol20 = info.get('vol20')
            vol_str = f"{vol20:.2%}" if vol20 is not None else 'N/A'
            vol_pct = info.get('vol_pct')
            vol_pct_str = f"分位{vol_pct:.0%}" if vol_pct is not None else '分位N/A'
            dist = info.get('dist_60high')
            dist_str = f"{dist:+.1%}" if dist is not None else 'N/A'
            lines.append(
                f"  {name}: MA60{ma_str}(连续{info.get('ma60_days', 0)}天) | "
                f"距60日高{dist_str} | "
                f"波动率{vol_str}({info.get('vol_state', '?')}, {vol_pct_str})")
        wr20 = dash.get('weighted_ret_20d')
        wr20_str = f"{wr20:+.2%}" if wr20 is not None else 'N/A'
        lines.append(f"  加权20日涨跌幅（成长加权）: {wr20_str}")

        sample = dash.get('sample', {})
        n_wide = sample.get('n_wide', 0)
        lines.append(f"  样本量: 宽口径 n={n_wide} | 严格口径 n={sample.get('n_strict', 0)}")

        stats = dash.get('stats')
        if stats:
            p5 = stats.get('cyb_5d_win_rate')
            lo5, hi5 = stats.get('cyb_5d_ci', (None, None))
            if p5 is not None and lo5 is not None:
                line = f"  创业板5日胜率: {p5:.1%} (95% CI [{lo5:.1%}, {hi5:.1%}])"
                uncond = stats.get('cyb_5d_uncond')
                if uncond:
                    line += f" vs 无条件基准 {uncond:.1%}"
                lines.append(line)
            p20 = stats.get('cyb_20d_win_rate')
            lo20, hi20 = stats.get('cyb_20d_ci', (None, None))
            if p20 is not None and lo20 is not None:
                lines.append(f"  创业板20日胜率: {p20:.1%} (95% CI [{lo20:.1%}, {hi20:.1%}])")
            m5 = stats.get('cyb_5d_mean')
            mlo, mhi = stats.get('cyb_5d_mean_ci', (None, None))
            if m5 is not None and mlo is not None:
                lines.append(f"  创业板5日均值: {m5:+.2%} (95% CI [{mlo:+.2%}, {mhi:+.2%}])")
        else:
            lines.append(f"  ⚠️ 样本量 n={n_wide}<{PSM_MIN_SAMPLE}，"
                         "不输出概率与胜率区间——仅定性跟踪")
        lines.append("")

        # ── 三剧本（定性结构；概率仅样本充足时展示）──
        probs_ok = scenarios.get('probabilities_available', False)
        lines.append("【三剧本结构】")
        for key in ['scenario_1', 'scenario_2', 'scenario_3']:
            s = scenarios.get(key, {})
            prob = s.get('probability')
            if probs_ok and prob is not None:
                bar = '█' * int(prob * 40)
                lines.append(f"  {s.get('name','')}: {prob:.0%} ({s.get('probability_range','')}) {bar}")
            else:
                lines.append(f"  {s.get('name','')}: —（{s.get('probability_range','样本不足')}）")
            lines.append(f"    触发: {'; '.join(s.get('trigger_conditions',[])[:2])}")
        lines.append("")

        # 历史回测（匹配次数引用；胜率区间已并入上方仪表盘）
        wide = bt.get('wide', {})
        strict = bt.get('strict', {})
        lines.append(f"【历史回测】严格口径: {strict.get('strict_matches',0)}次 | "
                     f"宽口径: {wide.get('wide_matches', 0)}次")
        lines.append("")

        # 波浪结构
        lines.append(f"【波浪结构】{wave.get('turnaround_judgment', 'N/A')}")
        lines.append("")

        # 关键点位
        levels = result.get('key_levels', {})
        for name in ['创业板指', '科创50', '上证指数']:
            if name in levels:
                lv = levels[name]
                res = lv.get('key_resistance', [])[:2]
                sup = lv.get('key_support', [])[:2]
                lines.append(f"  {name}: 压力{res} | 支撑{sup}")

        lines.append("")
        lines.append("⚠️ 不按预测交易，按确认交易。日线确认条件见 confirmation_rules。")
        lines.append("⚠️ 基于量化模型自动生成，仅供参考，不构成投资建议。")

        return '\n'.join(lines)
