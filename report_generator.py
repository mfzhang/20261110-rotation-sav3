#!/usr/bin/env python3
# /tmp/sav3/report_generator.py
# 报告 Markdown 生成器 — 每个报告板块为独立方法

import logging
from typing import Optional

import pandas as pd
import numpy as np

from config import CRISIS_EXPOSURE, CRISIS_STOCK_STOPLOSS
from ferm_hitrate_analyzer import format_hitrate_line

logger = logging.getLogger("report_generator")


class ReportGenerator:
    """每日量化参考报告的 Markdown 生成器，每个板块对应一个独立方法。"""

    def __init__(
        self,
        target_date: str,
        weekday: str,
        signals,
        crisis_mode: str,
        weights_used: dict,
        rolling_ic: Optional[dict],
        ranked: Optional[pd.DataFrame],
        prev_top20: dict,
        top_n: int,
        health: dict,
        stoploss_stocks: list,
        holdings_advice: list,
        top3_raw: list,
        bottom3_raw: list,
        tid_picks: list,
        perf: Optional[pd.DataFrame],
        ferment_result: Optional[dict],
        psm_cross_mark: str,
        psm_adjusted: Optional[dict],
        mt_result: Optional[dict],
        psm_linkage_enabled: bool = True,
        system_health: Optional[dict] = None,
        nav_perf: Optional[dict] = None,
        ferm_hitrate: Optional[dict] = None,
    ):
        self.target_date = target_date
        self.weekday = weekday
        self.signals = signals
        self.crisis_mode = crisis_mode
        self.weights_used = weights_used or {}
        self.rolling_ic = rolling_ic
        self.ranked = ranked
        self.prev_top20 = prev_top20
        self.top_n = top_n
        self.health = health
        self.stoploss_stocks = stoploss_stocks
        self.holdings_advice = holdings_advice
        self.top3_raw = top3_raw
        self.bottom3_raw = bottom3_raw
        self.tid_picks = tid_picks
        self.perf = perf
        self.ferment_result = ferment_result
        self.psm_cross_mark = psm_cross_mark
        self.psm_adjusted = psm_adjusted
        self.mt_result = mt_result
        self.psm_linkage_enabled = psm_linkage_enabled
        self.system_health = system_health or {}
        self.nav_perf = nav_perf
        self.ferm_hitrate = ferm_hitrate

    # ═══════════════════════════════════════════════
    # 报告板块方法
    # ═══════════════════════════════════════════════

    def header(self) -> str:
        return f"# 📈 每日量化参考 | {self.target_date} 周{self.weekday}\n\n"

    def nav_section(self) -> str:
        """纸面净值追踪：策略(Top50等权日频) vs 沪深300 的近 1/3/12 个月收益对比。

        数据来自 nav_history.json（每日选股后追加）；历史不足的窗口显示"积累中"。
        口径：收盘价比价、不含交易成本、未复权（与系统回测口径一致）。
        """
        if not self.nav_perf:
            return ""

        def _fmt(v, cov=None):
            if v is not None:
                return f"{v:+.1%}"
            if cov:
                return f"积累中({cov[0]}/{cov[1]}日)"
            return "—"

        strat = self.nav_perf.get('strategy', {})
        bench = self.nav_perf.get('benchmark', {})
        cov = self.nav_perf.get('coverage', {})
        windows = [('1m', '近1月'), ('3m', '近3月'), ('12m', '近12月')]

        f = []
        f.append("**📈 纸面净值追踪** "
                 f"（ARC Top50 等权·日频再平衡·不含成本，截至 {self.nav_perf.get('as_of', '—')}）\n\n")
        f.append("| | " + " | ".join(label for _, label in windows) + " |\n")
        f.append("|---|" + ":---:|" * len(windows) + "\n")
        f.append("| 策略 | " + " | ".join(_fmt(strat.get(k), cov.get(k)) for k, _ in windows) + " |\n")
        f.append("| 沪深300 | " + " | ".join(_fmt(bench.get(k)) for k, _ in windows) + " |\n")
        excess_cells = []
        for k, _ in windows:
            s, b = strat.get(k), bench.get(k)
            excess_cells.append(f"{s - b:+.1%}" if (s is not None and b is not None) else "—")
        f.append("| 超额 | " + " | ".join(excess_cells) + " |\n\n")
        return "".join(f)

    def risk_section(self) -> str:
        """风控状态"""
        idx_str = f"沪深300: {self.signals.index_latest:.1f}" if self.signals.index_latest else ""
        regime = getattr(self.signals, 'regime', '')
        regime_label = {'crash': '🔻崩盘/超跌', 'trend': '📈趋势市',
                        'neutral': '➖震荡'}.get(regime, '')
        if self.crisis_mode == 'crisis':
            lines = [
                "## 🛡️ ARC 风控: ⚠️ 危机模式\n\n",
                f"全仓 low_risk，仓位降至 {CRISIS_EXPOSURE:.0%}。{idx_str}\n\n",
            ]
            det = getattr(self.signals, 'detector', None)
            if det is not None and getattr(det, 'entry_date', None):
                lines.append(
                    f"危机触发: {det.entry_date.date()} ({det.trigger_reason or '原因未记录'})\n\n")
            if self.signals.need_rebalance:
                lines.append("⚠️ **紧急调仓**：危机信号首次触发，需要立即调整仓位至防御配置！\n\n")
            return "".join(lines)
        regime_str = f"市场状态: {regime_label} | " if regime_label else ""
        return (
            f"## 🛡️ ARC 风控: ✅ 正常模式\n\n"
            f"{regime_str}因子权重(动态): low_risk={self.weights_used.get('low_risk', 0):.0%}, "
            f"position={self.weights_used.get('position', 0):.0%}。{idx_str}\n\n"
        )

    def ic_section(self) -> str:
        """因子 IC 监控（月度20日前瞻 RankIC，与月度调仓周期对齐）"""
        if not self.rolling_ic:
            return ""
        ic_parts = []
        for fac in ['low_risk', 'position']:
            info = self.rolling_ic.get(fac, {})
            n_valid = info.get('n_valid', 0)
            if n_valid >= 12:
                ic_parts.append(f"{fac} IC={info['ic_mean']:+.2f} IR={info['ic_ir']:.2f}")
            elif n_valid > 0:
                ic_parts.append(f"{fac} N/A (数据积累中 {n_valid}/12期)")
            else:
                ic_parts.append(f"{fac} N/A (无数据)")
        return f"📊 因子月度20日 RankIC (对齐月度调仓周期, 近12期): {' | '.join(ic_parts)}\n\n"

    def arc_section(self) -> str:
        """ARC 月度选股"""
        f = []
        f.append("## 🏗️ ARC 月度选股\n\n")
        ranked = self.ranked
        if ranked is None or ranked.empty:
            f.append("选股数据未就绪。\n\n")
            return "".join(f)

        top_list = ranked.head(self.top_n)

        # 排名变化
        if self.prev_top20:
            prev_set = set(self.prev_top20)
            new_entries = [c for c in top_list.index[:5] if c not in prev_set]
            if new_entries:
                names = [top_list.loc[c, 'name'] for c in new_entries if c in top_list.index]
                f.append(f"**🆕 新进 Top5:** {', '.join(names)}\n\n")

        # Top 10 详细表
        f.append("### Top 10\n\n")
        f.append("| # | 代码 | 名称 | 行业 | 综合 | low_risk | position | 今收 | 建议买入价 |\n")
        f.append("|---|------|------|------|:---:|:-------:|:--------:|-----:|----------:|\n")

        for _, row in top_list.head(10).iterrows():
            code_short = row.name.split('.')[0] if '.' in str(row.name) else row.name
            close_val = row.get('close', None)
            buy_price = f"¥{close_val*0.99:.2f}" if close_val and not pd.isna(close_val) else "—"
            close_str = f"¥{close_val:.2f}" if close_val and not pd.isna(close_val) else "—"
            lr = f"{row['low_risk_rank']:.0%}" if 'low_risk_rank' in row and not pd.isna(row['low_risk_rank']) else \
                 (f"{row['low_risk']:+.2f}" if 'low_risk' in row and not pd.isna(row['low_risk']) else "—")
            pos = f"{row['position_rank']:.0%}" if 'position_rank' in row and not pd.isna(row['position_rank']) else \
                  (f"{row['position']:+.2f}" if 'position' in row and not pd.isna(row['position']) else "—")
            f.append(
                f"| {int(row['rank'])} | {code_short} | {row.get('name', '')} | "
                f"{row.get('industry', '')} | {row['composite_score']:.2f} | "
                f"{lr} | {pos} | {close_str} | {buy_price} |\n"
            )
        f.append("\n")

        # Top 11-top_n 简要
        if self.top_n > 10:
            f.append(f"<details>\n<summary>Top 11-{self.top_n} 展开</summary>\n\n")
            f.append("| # | 代码 | 名称 | 行业 | 综合得分 | 建议买入价 |\n")
            f.append("|---|------|------|------|:--------:|----------:|\n")
            for _, row in top_list.iloc[10:].iterrows():
                code_short = row.name.split('.')[0] if '.' in str(row.name) else row.name
                close_val = row.get('close', None)
                buy_price = f"¥{close_val*0.99:.2f}" if close_val and not pd.isna(close_val) else "—"
                f.append(
                    f"| {int(row['rank'])} | {code_short} | {row.get('name', '')} | "
                    f"{row.get('industry', '')} | {row['composite_score']:.2f} | {buy_price} |\n"
                )
            f.append("\n</details>\n\n")

        # HHI 健康指标
        if self.health.get('hhi') is not None:
            hhi_status = "⚠️ 偏高" if self.health['hhi'] > 0.15 else "✅ 分散"
            f.append(
                f"**行业集中度 HHI:** {self.health['hhi']:.3f} ({hhi_status}) | "
                f"最大行业: {self.health['max_industry']} ({self.health['max_industry_pct']:.0%})\n\n"
            )

        return "".join(f)

    def holdings_section(self) -> str:
        """持仓分析"""
        f = []
        f.append("## 💼 持仓\n\n")
        if self.stoploss_stocks:
            f.append(
                f"🚨 **个股止损警报**: {', '.join(self.stoploss_stocks)} 触发止损线 "
                f"({abs(CRISIS_STOCK_STOPLOSS):.0%})，建议立即清仓！\n\n"
            )
        if self.holdings_advice:
            f.append("| 代码 | 名称 | 成本 | 现价 | 持仓(股) | 盈亏(%) | 建议 |\n")
            f.append("|------|------|------|------|----------|--------|------|\n")
            for h in self.holdings_advice:
                cur_str = f"{h['current']:.2f}" if isinstance(h['current'], (int, float)) else str(h['current'])
                pct_str = f"{h['pct_change']:+.2f}" if h['pct_change'] is not None else 'N/A'
                f.append(
                    f"| {h['code']} | {h['name']} | {h['cost']} | {cur_str} | "
                    f"{h['shares']} | {pct_str} | {h['advice']} |\n"
                )
            f.append("\n")
        else:
            f.append("暂无持仓数据。创建 `holdings.json` 启用。\n\n")
        return "".join(f)

    def tid_section(self) -> str:
        """TID 短期选股 + 全部行业排名（20日夏普主信号，动量为对比参考）"""
        f = []
        f.append("## 🧭 TID 短期选股\n\n")
        f.append("> ⚠️ 以下为短期（20日行业夏普 + 个股短期强度）信号，与 ARC 月度选股独立。\n\n")

        # top3/bottom3 条目为 [行业名, 20日夏普]；动量从 perf 表查作对比参考
        def _fmt_ind(item):
            name, sharpe = item
            ref = ""
            if self.perf is not None and not self.perf.empty and 'return' in self.perf.columns:
                m = self.perf[self.perf['industry'] == name]
                if not m.empty:
                    ref = f", 动量{m.iloc[0]['return']:+.1f}%"
            return f"**{name}**(夏普{sharpe:+.2f}{ref})"

        if self.top3_raw:
            top_str = '  '.join([_fmt_ind(it) for it in self.top3_raw[:3]])
            f.append(f"**🔥 强势行业:** {top_str}\n\n")
        if self.bottom3_raw:
            bot_str = '  '.join([_fmt_ind(it) for it in self.bottom3_raw[:3]])
            f.append(f"**⚠️ 弱势行业:** {bot_str}\n\n")

        # TID 个股推荐
        if self.tid_picks:
            f.append("### 🎯 强势行业 Top5 个股\n\n")
            f.append("| 行业 | 代码 | 名称 | 20日动量(%) | 现价 | 建议买入价 | 建议止损价 |\n")
            f.append("|------|------|------|:----------:|-----:|:----------:|:----------:|\n")
            for p in self.tid_picks:
                mom_val = p['pct_chg']
                mom_str = f"{mom_val:+.1f}" if pd.notna(mom_val) else "—"
                close_val = p['close']
                close_str = f"¥{close_val:.2f}" if pd.notna(close_val) else "—"
                name_str = p['name'] if pd.notna(p.get('name')) else ''
                bp_val = p.get('buy_price')
                bp_str = f"¥{bp_val:.2f}" if pd.notna(bp_val) else "—"
                sl_val = p.get('stop_loss')
                sl_str = f"¥{sl_val:.2f}" if pd.notna(sl_val) else "—"
                f.append(
                    f"| {p['industry']} | {p['code']} | {name_str} | {mom_str} | {close_str} | {bp_str} | {sl_str} |\n"
                )
            f.append("\n")

        # 全部行业排名（折叠）
        if self.perf is not None and not self.perf.empty:
            f.append("<details>\n<summary>全部行业排名（按20日夏普，动量为参考）</summary>\n\n")
            f.append("| 行业 | 20日夏普 | 20日动量(%) | 成分股 |\n")
            f.append("|------|:--------:|:-----------:|:-----:|\n")
            for _, row in self.perf.iterrows():
                sharpe_str = f"{row['sharpe']:+.2f}" if pd.notna(row.get('sharpe')) else "—"
                f.append(f"| {row['industry']} | {sharpe_str} | {row['return']:+.2f} | {int(row['stocks'])} |\n")
            f.append("\n</details>\n\n")

        return "".join(f)

    def fermentation_section(self) -> str:
        """低位发酵观察"""
        if not self.ferment_result or self.ferment_result['fermentation'].empty:
            return ""

        ferm_df = self.ferment_result['fermentation']
        f = []
        f.append("## 🔥 低位发酵观察\n\n")
        f.append(
            "> ⚠️ 以下为「低位异动」信号——找的是有放量但未拥挤的子方向，与 TID 追涨信号互补。\n"
        )
        f.append(
            f"> PSM 市场水温：{self.psm_cross_mark} | ARC验证：ARC持仓在行业>50分位→✅ | "
            f"PSM验证：bull>60%→✅ bear>40%→⚠️\n"
        )
        f.append("> 数据源：日线 + 凌晨新闻，开盘前更新。\n\n")

        # 分档命中率（ferm_hitrate_analyzer 每日更新，次日子方向收益>0 为命中）
        hitrate_line = format_hitrate_line(self.ferm_hitrate) if self.ferm_hitrate else ""
        if hitrate_line:
            f.append(f"> 📊 {hitrate_line}\n\n")

        llm = self.ferment_result.get('llm_enhance', {})

        for tier_label, tier_desc, tier_emoji in [
            ('A', '低位发酵确认 — 可重点关注', '🟢'),
            ('B', '发酵偏强等确认 — 列入观察', '🟡'),
            ('C', '待触发 — 保持跟踪', '⚪'),
        ]:
            tier_df = ferm_df[ferm_df['tier'] == tier_label]
            if tier_df.empty:
                continue

            f.append(f"### {tier_emoji} {tier_label}档：{tier_desc}\n\n")
            f.append(
                "| 子方向 | 所属行业 | 52周位置 | 量比 | 20日涨幅 | 广度 | ARC验证 | PSM验证 | 催化判断 | 确认条件 | 失效条件 |\n"
            )
            f.append(
                "|--------|----------|:--------:|:----:|:--------:|:----:|:-------:|:-------:|----------|----------|----------|\n"
            )
            for _, row in tier_df.iterrows():
                sub = row['sub_sector']
                enh = llm.get(sub, {})
                catalyst = enh.get('catalyst', '—')
                confirm = enh.get('confirm_condition', '—')
                fail = enh.get('fail_condition', '—')
                arc_ok = row.get('arc_confirm', None)
                if arc_ok is True:
                    arc_str = "✅"
                elif arc_ok is False:
                    arc_str = "⚠️"
                else:
                    arc_str = "—"
                f.append(
                    f"| {sub} | {row['shenwan']} | {row['position']:.0%} | "
                    f"{row['volume_ratio']:.1f}x | {row['return_20d']:+.1%} | "
                    f"{row['breadth_3d']:.0%} | {arc_str} | {self.psm_cross_mark} | {catalyst} | {confirm} | {fail} |\n"
                )
            f.append("\n")

        # 已排除的拥挤方向
        excluded = self.ferment_result.get('excluded_crowded', [])
        if excluded:
            f.append("<details>\n<summary>❌ 已排除：拥挤方向</summary>\n\n")
            f.append("| 子方向 | 拥挤度 | 20日涨幅 |\n")
            f.append("|--------|:------:|:--------:|\n")
            for ex in excluded[:10]:
                f.append(
                    f"| {ex['sub_sector']} | {ex['crowding_score']:.2f} | {ex['return_20d']:+.1%} |\n"
                )
            f.append("\n</details>\n\n")

        return "".join(f)

    def psm_section(self) -> str:
        """PSM 指标仪表盘（2026-07-19 起：不输出精确概率，n<20 时仅定性跟踪）"""
        if not self.mt_result or not self.mt_result.get('summary'):
            return ""

        f = []
        f.append("## 🎯 PSM 市场仪表盘\n\n")
        f.append(
            "> ⚠️ PSM = Probabilistic Scenario Model（情景模型）。"
            "2026-07-19 起改为指标仪表盘模式：输出 MA60状态/距60日高点/波动率状态/"
            "加权20日涨跌幅/样本量/置信区间；历史匹配样本 n<20 时不输出概率，"
            "避免小样本精确感。不按预测交易，按确认交易。\n\n"
        )
        # P0-3 防线：指数数据缺失时必须显著标注（宁可缺板块，不可静默错）
        missing = self.mt_result.get('data_missing') or []
        if missing:
            f.append(f"> 🔴 **PSM 数据缺失**: {', '.join(missing)} —— "
                     f"以下指标基于不完整数据，结论可能失真！\n\n")
        f.append(self.mt_result['summary'])
        f.append("\n\n")

        # Q3: 展示级概率修正（仅 PSM 概率可用时展示；仪表盘模式 n<20 时跳过）
        if self.psm_linkage_enabled and self.psm_adjusted:
            orig = self.psm_adjusted['original']
            adj = self.psm_adjusted['adjusted']
            sig = self.psm_adjusted.get('signals', {})
            hhi_str = f"{sig.get('arc_hhi'):.3f}" if sig.get('arc_hhi') is not None else 'N/A'
            pos_str = f"{sig.get('arc_avg_position'):.1%}" if sig.get('arc_avg_position') is not None else 'N/A'
            breadth_str = f"{sig.get('tid_breadth'):.1%}" if sig.get('tid_breadth') is not None else 'N/A'
            disp_str = f"{sig.get('tid_dispersion'):.1%}" if sig.get('tid_dispersion') is not None else 'N/A'
            f.append("> **[PSM 联动]** 概率修正：\n")
            f.append("> - PSM 原始概率："
                    f"逼空延续 {orig['scenario_1']:.0%} | 直接逼空 {orig['scenario_2']:.0%} | "
                    f"兑现失败 {orig['scenario_3']:.0%}\n")
            f.append("> - 经 ARC/TID 修正概率："
                    f"逼空延续 {adj['scenario_1']:.0%} | 直接逼空 {adj['scenario_2']:.0%} | "
                    f"兑现失败 {adj['scenario_3']:.0%}\n")
            f.append("> - 修正输入："
                    f"ARC HHI={hhi_str}, ARC 平均位置={pos_str}, "
                    f"TID 广度={breadth_str}, TID Dispersion={disp_str}\n\n")

        # 确认规则折叠
        rules = self.mt_result.get('confirmation_rules', {})
        if rules.get('current_status'):
            f.append("<details>\n<summary>📋 日线确认规则</summary>\n\n")
            f.append("| 指数 | K线 | MA5 |\n")
            f.append("|------|:---:|:---:|\n")
            for name, status in rules['current_status'].items():
                yang = '🟢阳' if status.get('yang_xian') else '🔴阴'
                above = '✅站上' if status.get('above_ma5') else '❌跌破'
                f.append(f"| {name} | {yang} | {above} |\n")
            f.append("\n</details>\n\n")

        return "".join(f)

    def health_alert_banner(self) -> str:
        """系统健康红旗横幅（有红旗时置于报告顶部，防静默损坏）"""
        flags = self.system_health.get('red_flags') or []
        if not flags:
            return ""
        lines = ["## 🚩 系统健康警报\n\n"]
        for f in flags:
            lines.append(f"- 🔴 {f}\n")
        lines.append("\n> 以上问题可能使本报告部分内容失真，请先排查数据管线。\n\n")
        return "".join(lines)

    def system_health_section(self) -> str:
        """系统健康面板：各数据源行数/最新日期、因子 NaN 率、缓存规模、危机溯源。

        本次 4 个 P0（mdd NaN / 危机状态污染 / 指数限流截断 / 显示缩小100倍）
        的共性是静默——本面板是成本最低的总防线。
        """
        sh = self.system_health
        if not sh:
            return ""
        f = []
        f.append("## 🩺 系统健康\n\n")
        f.append("| 检查项 | 数值 | 最新日期 | 状态 |\n")
        f.append("|--------|------|----------|:----:|\n")

        sd = sh.get('stock_daily')
        if sd:
            f.append(f"| 个股日K ({sd['codes']}只) | {sd['rows']:,} 行 | {sd['latest']} | ✅ |\n")

        idx = sh.get('index_hs300')
        if idx:
            f.append(f"| 沪深300指数 | {idx['rows']} 行 | {idx['latest']} | ✅ |\n")

        nan_rates = sh.get('factor_nan') or {}
        for fac, rate in nan_rates.items():
            ok = rate <= 0.05
            mark = '✅' if ok else '🔴'
            f.append(f"| 因子 {fac} NaN率 | {rate:.1%} | {sh.get('factor_latest', '—')} | {mark} |\n")

        psm = sh.get('psm_indices') or {}
        if psm:
            for name, info in psm.items():
                if info.get('rows', 0) > 0:
                    f.append(f"| PSM {name} | {info['rows']} 行 | {info['latest']} | ✅ |\n")
                else:
                    f.append(f"| PSM {name} | 0 行 | — | 🔴 缺失 |\n")

        crisis = sh.get('crisis') or {}
        if crisis:
            mode = crisis.get('mode', '?')
            if mode == 'crisis':
                src = f"触发于 {crisis.get('entry_date')} ({crisis.get('trigger_reason') or '原因未记录'})" \
                    if crisis.get('entry_date') else "🔴 来源不明（无 entry_date）"
                f.append(f"| 危机状态 | {mode} | {src} | ⚠️ |\n")
            else:
                f.append(f"| 危机状态 | {mode} | — | ✅ |\n")

        if sh.get('cache_mb') is not None:
            f.append(f"| 缓存规模 | {sh['cache_mb']:.0f} MB | — | ✅ |\n")

        f.append("\n")
        return "".join(f)

    def disclaimer(self) -> str:
        """免责与说明"""
        return (
            "---\n"
            "> 因子模型: low_risk(低波动/低回撤/低Beta) + position(52周相对低位)，"
            "权重按市场状态动态调整（崩盘加权 position / 趋势市降权，见风控区）\n"
            f"> 危机防御: 沪深300 20日跌超{abs(0.15):.0%}或波动率跳升3倍\n"
            "> ⚠️ 基于量化模型自动生成，仅供参考，不构成投资建议。\n"
            "> ⚠️ 已知局限：行业分类为静态快照（来自最新 stock_basic），存在前视偏差风险。\n"
            ">    敏感性分析：5%标签扰动下年化收益波动 ±2.1%，夏普波动 ±0.06，不影响模型方向判断。\n"
        )

    # ═══════════════════════════════════════════════
    # 生成入口
    # ═══════════════════════════════════════════════

    def generate(self, report_path: str) -> str:
        """生成完整 Markdown 报告并写入文件，返回文件路径。"""
        sections = [
            self.header(),
            self.nav_section(),
            self.health_alert_banner(),
            self.risk_section(),
            self.ic_section(),
            self.arc_section(),
            self.holdings_section(),
            self.tid_section(),
            self.fermentation_section(),
            self.psm_section(),
            self.system_health_section(),
            self.disclaimer(),
        ]

        content = "".join(sections)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"✅ 报告已生成: {report_path}")
        return report_path
