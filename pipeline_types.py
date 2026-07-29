#!/usr/bin/env python3
# /tmp/sav3/pipeline_types.py
# 因子流水线共享类型定义

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from crisis_detector import CrisisDetector


@dataclass
class FactorPipelineResult:
    """
    ARC 因子流水线输出结构，替代原来的 signals dict。

    Attributes:
        crisis_mode: 'crisis' | 'normal'
        need_rebalance: 是否触发紧急调仓
        weights_used: 本次使用的因子权重 dict
        index_latest: 指数最新收盘价
        daily_df: 个股日K MultiIndex(date, code) DataFrame
        industry_map: Series(code -> industry)
        detector: CrisisDetector 实例，供持仓止损检查
        factors_df: 因子截面数据，供 IC 计算
    """
    crisis_mode: str = 'unknown'
    need_rebalance: bool = False
    weights_used: dict = field(default_factory=dict)
    index_latest: Optional[float] = None
    daily_df: Optional[pd.DataFrame] = None
    industry_map: Optional[pd.Series] = None
    detector: Optional[CrisisDetector] = None
    factors_df: Optional[pd.DataFrame] = None
    regime: str = ''           # 市场状态: crash/trend/neutral/crisis（动态配权依据）
    index_df: Optional[pd.DataFrame] = None  # 基准指数日K，供健康区/回测复用
    raw_factors_df: Optional[pd.DataFrame] = None  # pipeline_process 前的原始因子面板
                                                   # （TID×ARC 交叉验证复用，保持 position [0,1] 语义）
