#!/usr/bin/env python3
# /tmp/sav3/index_data.py
# 指数数据获取与预处理模块

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_fetcher import fetch_indices_daily

logger = logging.getLogger("index_data")

# ── 六大指数 tushare 代码 ──
INDEX_CODES = {
    '上证指数':   '000001.SH',
    '深证成指':   '399001.SZ',
    '创业板指':   '399006.SZ',
    '科创50':    '000688.SH',
    '沪深300':   '000300.SH',
    '中证1000':  '000852.SH',
}


class IndexDataFetcher:
    """指数数据拉取与统一数据表准备"""

    def __init__(self, end_date: str = None, start_date: str = '2015-01-01'):
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date

        # 懒加载数据容器
        self._raw_data: Dict[str, pd.DataFrame] = {}
        self._daily_returns: pd.DataFrame = None   # date × index_name
        self._daily_close: pd.DataFrame = None

    def fetch_all_indices(self, refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """拉取全部六大指数日K数据，统一走 data_fetcher。"""
        if self._raw_data and not refresh:
            return self._raw_data

        fields = ['open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'vol', 'amount']
        self._raw_data = fetch_indices_daily(
            self.start_date, self.end_date, INDEX_CODES, fields=fields, refresh=refresh
        )
        logger.info(f"  指数数据: {len(self._raw_data)}/{len(INDEX_CODES)} 个指数")
        return self._raw_data

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """准备统一数据表: daily_returns, daily_close (date × index_name)"""
        if self._daily_returns is not None:
            return self._daily_returns, self._daily_close

        self.fetch_all_indices()

        closes = {}
        returns = {}
        for name, df in self._raw_data.items():
            closes[name] = df['close']
            if 'pct_chg' in df.columns:
                returns[name] = df['pct_chg'] / 100.0  # 转小数
            else:
                returns[name] = df['close'].pct_change()

        self._daily_close = pd.DataFrame(closes).sort_index()
        self._daily_returns = pd.DataFrame(returns).sort_index()

        # 截取到 end_date
        end_ts = pd.Timestamp(self.end_date)
        self._daily_close = self._daily_close[self._daily_close.index <= end_ts]
        self._daily_returns = self._daily_returns[self._daily_returns.index <= end_ts]

        logger.info(f"  统一数据: {len(self._daily_close)} 个交易日, "
                     f"{len(self._daily_close.columns)} 个指数, "
                     f"范围 {self._daily_close.index[0].date()} ~ {self._daily_close.index[-1].date()}")
        return self._daily_returns, self._daily_close
