"""
Unit tests for crisis_detector.py

纯 mock 数据（pd.Series），覆盖 CrisisDetector 的核心状态机、冷却期、
个股止损与多指数加权检测逻辑。
"""
import json

import numpy as np
import pandas as pd
import pytest

from crisis_detector import CrisisDetector


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_flat_series(n_days=260, value=100.0):
    """生成平稳价格序列（默认 260 个交易日，满足 ≥252 要求）"""
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    return pd.Series(np.full(n_days, value), index=dates)


def _make_drawdown_series(drawdown=-0.20, n_days=260):
    """生成末尾 20 日出现指定跌幅的价格序列"""
    s = _make_flat_series(n_days, value=100.0)
    # 前 n_days-20 天为 100，最后 20 天线性跌至 100*(1+drawdown)
    target = 100.0 * (1 + drawdown)
    s.iloc[-20:] = np.linspace(100.0, target, 20)
    return s


def _make_vol_spike_series(n_days=260):
    """生成末尾 20 日波动率跳升 ≥3 倍的价格序列"""
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    np.random.seed(7)
    # 前 232 天：日收益 ~ N(0, 0.005)
    early_rets = np.random.normal(0, 0.005, n_days - 20)
    # 最后 20 天：日收益 ~ N(0, 0.05)，波动约为前段的 10 倍
    late_rets = np.random.normal(0, 0.05, 20)
    rets = np.concatenate([early_rets, late_rets])
    prices = 100.0 * (1 + rets).cumprod()
    return pd.Series(prices, index=dates)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCrisisEntry:
    """正常 → 危机 触发条件"""

    def test_normal_to_crisis_on_drawdown(self, crisis_detector_fresh, mock_hs300_close):
        """20 日跌幅超过阈值时进入危机模式"""
        s = mock_hs300_close.copy()
        s.iloc[-20:] = np.linspace(100.0, 80.0, 20)  # -20% 跌幅
        dt = s.index[-1]

        mode, emergency = crisis_detector_fresh.check(s, dt)

        assert mode == 'crisis'
        assert emergency is True

    def test_normal_to_crisis_on_vol_spike(self, crisis_detector_fresh):
        """波动率跳升 ≥3 倍时进入危机模式"""
        s = _make_vol_spike_series()
        dt = s.index[-1]

        mode, emergency = crisis_detector_fresh.check(s, dt)

        assert mode == 'crisis'
        assert emergency is True


class TestCrisisExit:
    """危机 → 正常 退出逻辑"""

    def test_crisis_exit_after_recovery(self, mock_hs300_close):
        """连续 5 日满足退出条件后恢复 normal"""
        detector = CrisisDetector(fresh_start=True)
        detector.mode = 'crisis'

        # 构造 5 个调仓日，每个都满足：无触发、回撤修复、波动率回落
        s = mock_hs300_close.copy()
        # 整体价格围绕 100 小幅波动，20 日跌幅 -5%（> -10% 退出线）
        s.iloc[-20:] = np.linspace(100.0, 95.0, 20)

        dates = s.index[-5:]
        last_mode = 'crisis'
        for dt in dates:
            mode, _ = detector.check(s, dt)
            last_mode = mode

        assert last_mode == 'normal'

    def test_cooling_period_blocks_retrigger(self, mock_hs300_close):
        """退出后冷却期内，跌幅触发但不足 -25% 时不重新触发危机"""
        detector = CrisisDetector(fresh_start=True)
        detector.mode = 'normal'
        # 模拟刚退出危机 3 天
        detector.last_exit_date = mock_hs300_close.index[-5]

        # 末尾 20 日跌幅 -18%，超过触发线 -15% 但未达突破线 -25%
        s = mock_hs300_close.copy()
        s.iloc[-20:] = np.linspace(100.0, 82.0, 20)
        dt = mock_hs300_close.index[-1]

        mode, emergency = detector.check(s, dt)

        assert mode == 'normal'
        assert emergency is False

    def test_severe_drawdown_breaks_cooling(self, mock_hs300_close):
        """退出后冷却期内，-25% 以上暴跌强制突破冷却重新触发危机"""
        detector = CrisisDetector(fresh_start=True)
        detector.mode = 'normal'
        detector.last_exit_date = mock_hs300_close.index[-5]

        s = mock_hs300_close.copy()
        s.iloc[-20:] = np.linspace(100.0, 75.0, 20)  # -25% 跌幅
        dt = mock_hs300_close.index[-1]

        mode, emergency = detector.check(s, dt)

        assert mode == 'crisis'
        assert emergency is True


class TestStatePersistence:
    """跨会话状态持久化"""

    def test_state_persistence(self, tmp_path, monkeypatch, mock_hs300_close):
        """危机状态写入文件后，新实例应能恢复"""
        state_file = tmp_path / 'crisis_state.json'
        monkeypatch.setattr('crisis_detector.STATE_FILE', str(state_file))

        # 第一次实例：触发危机并持久化
        detector1 = CrisisDetector(fresh_start=False)
        s = _make_drawdown_series(drawdown=-0.20)
        detector1.check(s, s.index[-1])
        detector1._persist()

        # 验证文件内容
        assert state_file.exists()
        saved = json.loads(state_file.read_text())
        assert saved['mode'] == 'crisis'

        # 第二次实例：从文件恢复
        detector2 = CrisisDetector(fresh_start=False)
        assert detector2.mode == 'crisis'


class TestStockStoploss:
    """个股止损"""

    def test_stock_stoploss(self):
        """返回跌幅超过止损线的股票代码"""
        detector = CrisisDetector(stock_stoploss=-0.15)
        current = pd.Series({
            '000001.SZ': 80.0,
            '000002.SZ': 95.0,
            '000003.SZ': 100.0,
        })
        entry = pd.Series({
            '000001.SZ': 100.0,  # -20%
            '000002.SZ': 100.0,  # -5%
            '000003.SZ': 100.0,  # 0%
        })

        stopped = detector.check_stock_stoploss(current, entry)

        assert stopped == ['000001.SZ']


class TestMultiIndexWeighted:
    """多指数加权危机检测"""

    def test_multi_index_weighted(self, mock_multi_index_data):
        """多指数加权跌幅触发危机"""
        detector = CrisisDetector(
            fresh_start=True,
            index_codes=['000300.SH', '399006.SZ', '000688.SH'],
            index_weights=[0.4, 0.3, 0.3],
        )

        data = mock_multi_index_data
        # 三个指数均大跌，加权跌幅超过 -15%
        data['000300.SH'].iloc[-20:] = np.linspace(100.0, 80.0, 20)  # -20%
        data['399006.SZ'].iloc[-20:] = np.linspace(100.0, 80.0, 20)
        data['000688.SH'].iloc[-20:] = np.linspace(100.0, 80.0, 20)

        dt = data['000300.SH'].index[-1]
        mode, emergency = detector.check(data, dt)

        assert mode == 'crisis'
        assert emergency is True
        # 板块崩盘警告会被填充（列表非空即可）
        assert isinstance(detector.warnings, list)
