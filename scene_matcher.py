#!/usr/bin/env python3
# /tmp/sav3/scene_matcher.py
# 历史场景相似度匹配

import logging
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("scene_matcher")

# 用于相似度计算的指数权重 (重点看成长/科技)
SIMILARITY_INDEX_WEIGHTS = {
    '创业板指': 0.35,
    '科创50':  0.35,
    '上证指数': 0.15,
    '中证1000': 0.15,
}


class StandardScaler:
    """手写 z-score 标准化器（不引入 sklearn 依赖）。

    fit 用 nan 统计容忍缺失特征；全缺失列均值置 0、零方差/缺失列标准差置 1（防除零）。
    transform 后 NaN 保留，调用方按需 nan_to_num（缺失 → 0 = 列均值，即"无信息"）。
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            # 全 NaN 列会触发 "All-NaN slice" RuntimeWarning，由下方兜底处理
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean_ = np.nanmean(X, axis=0)
            self.std_ = np.nanstd(X, axis=0)
        self.mean_ = np.where(np.isnan(self.mean_), 0.0, self.mean_)
        self.std_ = np.where(np.isnan(self.std_) | (self.std_ == 0), 1.0, self.std_)
        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler 尚未 fit")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SceneMatcher:
    """历史场景相似度匹配器"""

    def __init__(self):
        # 懒加载数据容器，由 MarketTiming 在委托前注入
        self._daily_returns: pd.DataFrame = None
        self._daily_close: pd.DataFrame = None

    def compute_state_vector(self, date: pd.Timestamp) -> Dict[str, float]:
        """计算某日的状态向量，用于相似度匹配"""
        returns, close = self._daily_returns, self._daily_close

        idx = returns.index.get_loc(date)
        if idx < 60:
            return None

        state = {}

        for name in ['创业板指', '科创50', '上证指数', '中证1000']:
            if name not in close.columns:
                continue
            s = close[name]
            current = s.iloc[idx]

            # 各周期涨跌幅
            for n, label in [(5, '5d'), (8, '8d'), (20, '20d'), (60, '60d')]:
                if idx >= n:
                    past = s.iloc[idx - n]
                    state[f'{name}_{label}'] = (current - past) / past

            # 距60日高点距离
            if idx >= 60:
                hh_60 = s.iloc[idx - 60:idx + 1].max()
                state[f'{name}_dist_60high'] = (current - hh_60) / hh_60

            # 距20日低点距离
            if idx >= 20:
                ll_20 = s.iloc[idx - 20:idx + 1].min()
                state[f'{name}_dist_20low'] = (current - ll_20) / ll_20

            # 波动率 (20日)
            if idx >= 20:
                vol = returns[name].iloc[idx - 19:idx + 1].std()
                state[f'{name}_vol20'] = vol

        # 成长相对上证超额
        for growth in ['创业板指', '科创50']:
            if growth in close.columns and '上证指数' in returns.columns:
                ret_g = returns[growth].iloc[idx - 7:idx + 1].sum() if idx >= 7 else 0
                ret_sh = returns['上证指数'].iloc[idx - 7:idx + 1].sum() if idx >= 7 else 0
                state[f'{growth}_excess_8d'] = ret_g - ret_sh

        return state

    def find_similar_scenes(self, current_date: pd.Timestamp = None,
                            top_n: int = 20, min_distance: int = 20) -> pd.DataFrame:
        """
        历史类比搜索: 找出历史上与当前状态最相似的场景。

        相似度计算前对特征做 z-score 标准化（在历史候选样本上 fit）：
        原始特征量纲差异巨大（60日涨跌幅 ±30% vs vol20 ~2%），
        不标准化时大量纲特征主导余弦相似度，小量纲特征形同虚设。

        Returns:
            DataFrame with columns: date, similarity, and subsequent N-day returns
        """
        returns, close = self._daily_returns, self._daily_close

        if current_date is None:
            current_date = close.index[-1]

        # 当前状态向量
        current_state = self.compute_state_vector(current_date)
        if current_state is None:
            logger.warning("无法计算当前状态向量 (数据不足)")
            return pd.DataFrame()

        feature_names = sorted(current_state.keys())

        # 特征权重向量（按所属指数，成长/科技权重高）
        w_vec = np.ones(len(feature_names))
        for j, feat in enumerate(feature_names):
            for idx_name, idx_w in SIMILARITY_INDEX_WEIGHTS.items():
                if feat.startswith(idx_name):
                    w_vec[j] = idx_w
                    break

        # 遍历历史，收集状态向量
        hist_idx = []     # close 中的位置（供后续涨跌幅计算）
        hist_dates = []
        hist_rows = []
        for i in range(60, len(close) - 21):  # 至少留20天给后续计算
            hist_date = close.index[i]
            if abs((hist_date - current_date).days) < min_distance:
                continue  # 跳过太近的日期

            hist_state = self.compute_state_vector(hist_date)
            if hist_state is None:
                continue

            hist_idx.append(i)
            hist_dates.append(hist_date)
            hist_rows.append([hist_state.get(feat, np.nan) for feat in feature_names])

        if not hist_rows:
            return pd.DataFrame()

        # z-score 标准化：在历史候选样本上 fit，同时变换当前向量
        H_raw = np.asarray(hist_rows, dtype=float)
        scaler = StandardScaler().fit(H_raw)
        H = np.nan_to_num(scaler.transform(H_raw), nan=0.0)
        c_vec = np.array([current_state.get(feat, np.nan) for feat in feature_names],
                         dtype=float)
        c_vec = np.nan_to_num(scaler.transform(c_vec.reshape(1, -1))[0], nan=0.0)

        # 加权余弦相似度（向量化）
        dot = (H * c_vec * w_vec).sum(axis=1)
        norm_c = np.sqrt((w_vec * c_vec * c_vec).sum())
        norm_h = np.sqrt((H * H * w_vec).sum(axis=1))
        denom = norm_c * norm_h
        sims = np.where(denom > 0, dot / np.maximum(denom, 1e-12), 0.0)

        # 计算后续N日涨跌幅
        results = []
        for k, i in enumerate(hist_idx):
            row = {'date': hist_dates[k], 'similarity': float(sims[k])}
            for name in close.columns:
                s = close[name]
                for n in [1, 2, 3, 5, 10, 20]:
                    if i + n < len(s):
                        fwd_ret = (s.iloc[i + n] - s.iloc[i]) / s.iloc[i]
                        row[f'{name}_fwd{n}d'] = fwd_ret
                    else:
                        row[f'{name}_fwd{n}d'] = np.nan
            results.append(row)

        df = pd.DataFrame(results)
        df = df.sort_values('similarity', ascending=False).head(top_n * 2)
        return df
