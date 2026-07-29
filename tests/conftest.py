"""共享 fixtures：合成数据，不调 tushare API"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def synthetic_daily_df():
    """合成 300 交易日 × 50 只股票的日K数据"""
    np.random.seed(42)
    n_days = 300
    n_stocks = 50
    dates = pd.date_range('2025-01-01', periods=n_days, freq='B')
    codes = [f'{i:06d}.SH' for i in range(n_stocks)]
    
    # 映射每只股票到一个子方向（确保 ≥3 只/方向）
    subs = ['储能', '光伏', '锂电池', '半导体', 'PCB/覆铜板', 'AI应用', '小金属']
    code_sub = {}
    for i, c in enumerate(codes):
        code_sub[c] = subs[i % len(subs)]
    
    rows = []
    for d in dates:
        for c in codes:
            rows.append({
                'date': d, 'code': c,
                'open': 100, 'high': 105, 'low': 95,
                'close': np.random.uniform(50, 150),
                'pct_chg': np.random.normal(0, 2),
                'amount': np.random.uniform(1e7, 1e8),
                'volume': np.random.uniform(1e5, 1e6),
                'name': code_sub[c] + '科技',
            })
    return pd.DataFrame(rows).set_index(['date', 'code'])


@pytest.fixture(scope="session")
def synthetic_industry_map():
    """code → 申万一级行业"""
    codes = [f'{i:06d}.SH' for i in range(50)]
    subs = ['储能', '光伏', '锂电池', '半导体', 'PCB/覆铜板', 'AI应用', '小金属']
    mapping = {
        '储能': '电力设备', '光伏': '电力设备', '锂电池': '电力设备',
        '半导体': '电子', 'PCB/覆铜板': '电子',
        'AI应用': '计算机',
        '小金属': '有色金属',
    }
    return pd.Series({c: mapping[subs[i % len(subs)]] for i, c in enumerate(codes)})


@pytest.fixture(scope="session")
def synthetic_ferm_df():
    """合成发酵检测输出 DataFrame"""
    return pd.DataFrame({
        'sub_sector': ['小金属', 'AI应用', '锂电池'],
        'shenwan': ['有色金属', '计算机', '电力设备'],
        'position': [0.65, 0.98, 0.55],
        'volume_ratio': [1.7, 1.0, 1.3],
        'return_20d': [0.199, -0.196, 0.05],
        'breadth_3d': [0.67, 0.0, 0.5],
        'ferment_score': [0.62, 0.45, 0.41],
        'tier': ['B', 'C', 'C'],
        'crowding_score': [0.5, 0.3, 0.6],
        'consecutive_up': [3, 1, 2],
        'turnover_rank_pct': [0.6, 0.4, 0.5],
        'up_down_ratio': [1.5, 0.8, 1.2],
        'n_stocks': [10, 8, 12],
    })


@pytest.fixture
def crisis_detector_fresh():
    """ fresh_start=True 的 CrisisDetector，避免污染持久化状态 """
    from crisis_detector import CrisisDetector
    return CrisisDetector(fresh_start=True)


@pytest.fixture
def mock_hs300_close():
    """ 沪深300 mock 收盘价序列：252 个交易日，默认平稳 """
    dates = pd.date_range('2024-01-01', periods=260, freq='B')
    # 默认价格=100，便于构造跌幅
    return pd.Series(100.0, index=dates)


@pytest.fixture
def mock_multi_index_data(mock_hs300_close):
    """ 多指数 mock 数据字典，兼容 CrisisDetector 多指数加权模式 """
    dates = mock_hs300_close.index
    return {
        '000300.SH': mock_hs300_close.copy(),
        '399006.SZ': pd.Series(100.0, index=dates),
        '000688.SH': pd.Series(100.0, index=dates),
    }
