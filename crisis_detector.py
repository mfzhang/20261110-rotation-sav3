# ARC 危机检测模块 — 含波动率确认 + 跨日状态持久化
import os
import json
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(os.path.dirname(__file__), 'cache', 'crisis_state.json')

def _load_state():
    """从磁盘加载危机检测器状态（跨 cron 会话持久化）"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            # 合理性校验：crisis 状态必须有 entry_date 溯源，否则来源不明
            # （历史上曾被 pytest/回测脚本污染，见 review_2026-07-18 §2.2）
            if state.get('mode') == 'crisis' and not state.get('entry_date'):
                logger.warning(
                    "⚠️ crisis_state.json 处于 crisis 但无 entry_date，状态来源不明"
                    "（疑似测试/回测污染）。请人工核实后重置为 normal 或补登 entry_date。")
            return state
        except (json.JSONDecodeError, IOError):
            pass
    return {'mode': 'normal', 'consecutive_normal': 0, 'last_exit_date': None,
            'entry_date': None, 'trigger_reason': None}

def _save_state(state: dict):
    """持久化状态到磁盘"""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, default=str)


class CrisisDetector:
    def __init__(self, index_drawdown=-0.15, vol_ratio=3.0,
                 exit_drawdown=-0.10, exit_vol_ratio=1.5, exit_days=5, cooling_days=10,
                 stock_stoploss=-0.15, fresh_start=False, persist=None,
                 index_codes=None, index_weights=None):
        # 从持久化状态恢复（跨 cron 会话）
        # fresh_start=True: 回测/测试专用，忽略持久化文件，从 normal 开始
        # persist: 是否允许写生产状态文件。默认 fresh_start=True 时禁用写入——
        #   回测/测试绝不允许污染生产 crisis_state.json（P0-2 根因修复：
        #   历史上根目录 pytest 与 run_*_backtest.py 均经 _persist() 写入生产状态）
        if persist is None:
            persist = not fresh_start
        self.persist = persist
        if fresh_start:
            self.mode = 'normal'
            self.consecutive_normal = 0
            self.last_exit_date = None
            self.entry_date = None
            self.trigger_reason = None
        else:
            saved = _load_state()
            self.mode = saved.get('mode', 'normal')
            self.consecutive_normal = saved.get('consecutive_normal', 0)
            self.last_exit_date = (pd.Timestamp(saved['last_exit_date'])
                                   if saved.get('last_exit_date') else None)
            self.entry_date = (pd.Timestamp(saved['entry_date'])
                               if saved.get('entry_date') else None)
            self.trigger_reason = saved.get('trigger_reason')

        self.idx_dd = index_drawdown
        self.vol_r = vol_ratio
        self.exit_dd = exit_drawdown
        self.exit_vol_r = exit_vol_ratio  # 新增：退出需波动率回落到 1.5x 以内
        self.exit_days = exit_days
        self.cooling_days = cooling_days
        self.stock_stoploss = stock_stoploss

        # P1修复：多指数加权危机检测
        # 默认单指数模式（向后兼容），传入 codes/weights 启用多指数加权
        self.index_codes = index_codes if index_codes is not None else ['000300.SH']
        self.index_weights = index_weights if index_weights is not None else [1.0]
        self.warnings = []  # 每次 check() 调用后填充的警告列表

    def _persist(self):
        """保存当前状态（persist=False 时为空操作——回测/测试不写生产文件）"""
        if not self.persist:
            return
        _save_state({
            'mode': self.mode,
            'consecutive_normal': self.consecutive_normal,
            'last_exit_date': str(self.last_exit_date.date()) if self.last_exit_date else None,
            'entry_date': str(self.entry_date.date()) if self.entry_date else None,
            'trigger_reason': self.trigger_reason,
        })

    def check(self, index_data, date):
        """检测当前是否触发/退出危机模式。返回 ('crisis'|'normal', 是否需要紧急调仓)

        P1修复：支持多指数加权检测。
        index_data: pd.Series (单指数，向后兼容) 或 dict {code: pd.Series} (多指数模式)。
        调用后可通过 self.warnings 获取板块崩盘警告列表。
        """
        self.warnings = []

        # --- 解析指数数据：dict 为多指数模式，Series 为旧单指数模式 ---
        if isinstance(index_data, dict):
            index_closes = index_data
        else:
            # 向后兼容：单指数模式，包装为 dict
            index_closes = {'000300.SH': index_data}

        # 沪深300 作为波动率基准（流动性最好），必须存在且有足够数据
        hs300 = index_closes.get('000300.SH')
        if hs300 is None or len(hs300) < 252:
            return self.mode, False

        # --- P1: 计算多指数加权跌幅 ---
        weighted_drop = 0.0
        total_weight = 0.0
        for code, weight in zip(self.index_codes, self.index_weights):
            series = index_closes.get(code)
            if series is not None and len(series) >= 20:
                drop = series.iloc[-1] / series.iloc[-20] - 1.0
                weighted_drop += weight * drop
                total_weight += weight
        if total_weight > 0:
            weighted_drop = weighted_drop / total_weight  # 归一化（应对部分指数缺失）

        # --- 沪深300 波动率基准 ---
        vol20_hs = hs300.pct_change().tail(20).std() * np.sqrt(252)
        vol250_hs = hs300.pct_change().tail(252).std() * np.sqrt(252)

        # --- P1: 创业板/科创50 独立波动率检测（板块崩盘警告）---
        sector_crash = False
        for code in ['399006.SZ', '000688.SH']:
            s = index_closes.get(code)
            if s is not None and len(s) >= 252:
                v20 = s.pct_change().tail(20).std() * np.sqrt(252)
                v250 = s.pct_change().tail(252).std() * np.sqrt(252)
                if v250 > 0 and v20 / v250 >= self.vol_r:
                    sector_crash = True
                    self.warnings.append(
                        f"sector_crash_warning: {code} vol spike "
                        f"({v20 / v250:.1f}x threshold={self.vol_r}x)"
                    )

        # --- 危机判断 ---
        drawdown_trigger = weighted_drop <= self.idx_dd
        vol_trigger = vol250_hs > 0 and vol20_hs / vol250_hs >= self.vol_r
        crisis_now = drawdown_trigger or vol_trigger

        # 板块崩盘但不满足全面危机条件：发出警告但不切 full crisis mode
        if sector_crash and not crisis_now:
            self.warnings.append(
                "sector_crash_warning: growth indices vol spike without HS300 confirmation, "
                "not entering full crisis mode"
            )

        if self.mode == 'normal' and crisis_now:
            # 冷却期检查：退出后10天内不重触发
            if self.last_exit_date is not None:
                days_since_exit = (date - self.last_exit_date).days
                if days_since_exit <= self.cooling_days:
                    # 超严厉阈值（-25%）：强制突破冷却，重触发危机
                    if weighted_drop <= -0.25:
                        pass  # fall through to re-trigger
                    else:
                        return 'normal', False
            self.mode = 'crisis'
            self.consecutive_normal = 0
            self.entry_date = pd.Timestamp(date)
            reasons = []
            if drawdown_trigger:
                reasons.append(f"加权20日跌幅{weighted_drop:+.1%}≤{self.idx_dd:.0%}")
            if vol_trigger:
                reasons.append(f"波动率比{vol20_hs / vol250_hs:.1f}x≥{self.vol_r:.0f}x")
            self.trigger_reason = '; '.join(reasons)
            self._persist()
            return 'crisis', True  # 紧急调仓！

        if self.mode == 'crisis':
            # 退出条件：不再有危机信号 + 加权回撤修复 + 波动率回落
            vol_normalized = vol250_hs > 0 and vol20_hs / vol250_hs < self.exit_vol_r
            exit_ok = (not crisis_now
                       and weighted_drop > self.exit_dd
                       and vol_normalized)
            if exit_ok:
                self.consecutive_normal += 1
                if self.consecutive_normal >= self.exit_days:
                    self.mode = 'normal'
                    self.last_exit_date = date
                    self.entry_date = None
                    self.trigger_reason = None
                    self._persist()
                    return 'normal', False
                self._persist()  # 退出计数中需持久化 consecutive_normal
            else:
                if self.consecutive_normal != 0:
                    self.consecutive_normal = 0
                    self._persist()  # 计数重置需持久化
            return 'crisis', False

        return 'normal', False

    def check_stock_stoploss(self, current_prices, entry_prices):
        """
        检查个股止损：返回需要平仓的股票列表。
        current_prices: Series(index=code, values=当前价格)
        entry_prices: Series(index=code, values=入场价格)
        返回: list of codes 触发止损的股票代码
        """
        if current_prices is None or entry_prices is None:
            return []
        if len(current_prices) == 0 or len(entry_prices) == 0:
            return []
        common = current_prices.index.intersection(entry_prices.index)
        if len(common) == 0:
            return []
        returns = current_prices[common] / entry_prices[common] - 1.0
        stopped = returns[returns <= self.stock_stoploss]
        return stopped.index.tolist()
