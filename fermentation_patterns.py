#!/usr/bin/env python3
# /tmp/sav3/fermentation_patterns.py
"""
发酵模式管理 — 确认/失效条件的系统化管理和回测验证

独立模块，负责：
1. 保存每天的发酵分档结果和 LLM 生成的确认/失效条件
2. 加载历史模式数据
3. 回测确认/失效条件的命中率
4. 筛选高命中率的可复用模式

存储格式: /tmp/sav3/output/ferm_patterns.json
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
from config import CACHE_DIR
from sub_sectors import classify_stock_to_sub

logger = logging.getLogger("fermentation_patterns")

# === 路径配置 ===
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
PATTERNS_FILE = os.path.join(OUTPUT_DIR, 'ferm_patterns.json')

# === 回测参数 ===
BACKTEST_RETURN_THRESHOLD = 0.0       # 次日收益 > 0 视为确认信号命中
BACKTEST_FAIL_THRESHOLD = -0.005      # 次日收益 <= -0.5% 视为失效信号命中
BACKTEST_VOL_RATIO_CONFIRM = 1.05     # 量比 > 1.05 辅助确认


def _ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_patterns(ferm_df, llm_enhance, date_str):
    """保存每天的发酵分档结果和 LLM 生成的确认/失效条件到 JSON 文件。

    输入:
      ferm_df: fermentation pipeline 输出的 DataFrame
               (含 tier, sub_sector, shenwan, position, volume_ratio 等)
      llm_enhance: dict {sub_sector: {catalyst, confirm_condition, fail_condition}}
      date_str: '2026-06-19'

    输出: 追加到 ferm_patterns.json
    """
    _ensure_output_dir()

    # 加载已有数据
    existing = {}
    if os.path.exists(PATTERNS_FILE):
        try:
            with open(PATTERNS_FILE, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning(f"{PATTERNS_FILE} 损坏，将重新创建")
            existing = {}

    # 构建当日记录
    records = []
    for _, row in ferm_df.iterrows():
        sub = row['sub_sector']
        enh = llm_enhance.get(sub, {}) if llm_enhance else {}

        record = {
            'sub_sector': sub,
            'shenwan': row.get('shenwan', ''),
            'tier': row.get('tier', ''),
            'position': round(float(row.get('position', 0)), 4),
            'volume_ratio': round(float(row.get('volume_ratio', 0)), 4),
            'return_20d': round(float(row.get('return_20d', 0)), 4),
            'breadth_3d': round(float(row.get('breadth_3d', 0)), 4),
            'breadth_momentum': round(float(row.get('breadth_momentum', 0)), 4),
            'consecutive_up': int(row.get('consecutive_up', 0)),
            'crowding_score': round(float(row.get('crowding_score', 0)), 4),
            'ferment_score': round(float(row.get('ferment_score', 0)), 4),
            'n_stocks': int(row.get('n_stocks', 0)),
            'catalyst': enh.get('catalyst', ''),
            'confirm_condition': enh.get('confirm_condition', ''),
            'fail_condition': enh.get('fail_condition', ''),
        }
        records.append(record)

    # 追加/覆盖当日数据
    existing[date_str] = records

    with open(PATTERNS_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    logger.info(f"已保存 {len(records)} 条发酵模式到 {PATTERNS_FILE} (日期: {date_str})")


def load_patterns(days=60):
    """加载最近 N 天的模式数据。

    返回: dict {date: [records]}
    """
    if not os.path.exists(PATTERNS_FILE):
        logger.info(f"{PATTERNS_FILE} 不存在，返回空字典")
        return {}

    try:
        with open(PATTERNS_FILE, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"读取 {PATTERNS_FILE} 失败: {e}")
        return {}

    if not all_data:
        return {}

    # 按日期排序，取最近 N 天
    sorted_dates = sorted(all_data.keys(), reverse=True)
    recent_dates = sorted(sorted_dates[:days])

    return {d: all_data[d] for d in recent_dates}


def _find_daily_cache_files():
    """查找可用的 daily_df 缓存文件，返回按日期范围排序的列表"""
    if not os.path.exists(CACHE_DIR):
        return []

    cache_files = []
    for fname in os.listdir(CACHE_DIR):
        if fname.startswith('ts_daily_') and fname.endswith('.pkl'):
            fpath = os.path.join(CACHE_DIR, fname)
            cache_files.append(fpath)

    # 按修改时间倒序
    cache_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cache_files


def _load_latest_daily_cache():
    """加载最新的 daily_df 缓存，返回 DataFrame 或 None"""
    cache_files = _find_daily_cache_files()
    for fpath in cache_files:
        try:
            df = pd.read_pickle(fpath)
            if df is not None and not df.empty:
                # 确保有必要的列
                required_cols = ['pct_chg', 'amount']
                if all(c in df.columns for c in required_cols):
                    logger.info(f"加载缓存: {os.path.basename(fpath)} ({len(df)} 行)")
                    return df
        except Exception as e:
            logger.debug(f"跳过 {fpath}: {e}")
            continue
    return None


def _get_next_trading_day(date_str, daily_df):
    """获取 date_str 的下一个交易日。

    从 daily_df 的索引中找。
    """
    target = pd.Timestamp(date_str)
    dates = sorted(daily_df.index.get_level_values('date').unique())
    # 找到 > target 的第一个日期
    for d in dates:
        if d > target:
            return d.strftime('%Y-%m-%d')
    return None


def _compute_sub_daily_return(daily_df, date_str):
    """计算某日各子方向的加权日收益。

    返回: {sub_sector: daily_return}
    """
    target_date = pd.Timestamp(date_str)
    try:
        day_data = daily_df.xs(target_date, level='date')
    except KeyError:
        return {}

    if day_data.empty:
        return {}

    # 分类每只股票
    sub_returns = {}  # {sub: [(ret, amt), ...]}
    for code, row in day_data.iterrows():
        name = str(row.get('name', '')) if 'name' in row.index else ''
        sub = classify_stock_to_sub(str(code), name)
        if sub is None:
            continue
        ret = row.get('pct_chg', 0) / 100.0
        amt = row.get('amount', 0)
        if pd.isna(ret) or pd.isna(amt):
            continue
        sub_returns.setdefault(sub, []).append((ret, amt))

    # 加权聚合
    result = {}
    for sub, items in sub_returns.items():
        if len(items) < 2:
            continue
        total_amt = sum(a for _, a in items)
        if total_amt <= 0:
            continue
        weighted_ret = sum(r * a for r, a in items) / total_amt
        result[sub] = weighted_ret

    return result


def _compute_sub_volume_ratio(daily_df, date_str):
    """计算某日各子方向的量比（当日成交额 / 前5日均成交额）。

    返回: {sub_sector: volume_ratio}
    """
    target_date = pd.Timestamp(date_str)

    # 需要 target_date 和前5个交易日
    all_dates = sorted(daily_df.index.get_level_values('date').unique())
    try:
        t_idx = all_dates.index(target_date)
    except ValueError:
        return {}

    lookback_dates = all_dates[max(0, t_idx - 5):t_idx + 1]

    # 聚合每个子方向每天的成交额
    sub_daily_amt = {}  # {sub: {date: total_amt}}
    for i, d in enumerate(lookback_dates):
        try:
            day_data = daily_df.xs(d, level='date')
        except KeyError:
            continue
        for code, row in day_data.iterrows():
            name = str(row.get('name', '')) if 'name' in row.index else ''
            sub = classify_stock_to_sub(str(code), name)
            if sub is None:
                continue
            amt = row.get('amount', 0)
            if pd.isna(amt):
                continue
            sub_daily_amt.setdefault(sub, {}).setdefault(d, 0)
            sub_daily_amt[sub][d] += amt

    # 计算量比
    result = {}
    for sub, date_amts in sub_daily_amt.items():
        sorted_dates = sorted(date_amts.keys())
        if len(sorted_dates) < 2:
            continue
        today_amt = date_amts.get(target_date, 0)
        prev_dates = [d for d in sorted_dates if d < target_date]
        if not prev_dates:
            continue
        prev_avg = np.mean([date_amts[d] for d in prev_dates[-5:]])
        if prev_avg > 0:
            result[sub] = today_amt / prev_avg
        else:
            result[sub] = 1.0

    return result


def backtest_conditions(days=60):
    """回测确认/失效条件的命中率。

    对每条 'confirm_condition'，检查次日该子方向是否满足（用次日涨跌幅、量比等简单判断）。
    对每条 'fail_condition'，同样检查。

    返回: DataFrame {sub_sector, confirm_hit_rate, fail_hit_rate, total_checks}
    如果历史数据不足，返回空 DataFrame。
    """
    patterns = load_patterns(days=days)
    if not patterns:
        logger.info("无历史模式数据，返回空 DataFrame")
        return pd.DataFrame()

    # 加载日线数据
    daily_df = _load_latest_daily_cache()
    if daily_df is None:
        logger.info("无日线缓存数据，返回空 DataFrame")
        return pd.DataFrame()

    # 收集所有检查结果
    # {sub_sector: {'confirm_hits': int, 'confirm_total': int, 'fail_hits': int, 'fail_total': int}}
    stats = {}

    for date_str, records in patterns.items():
        # 找下一个交易日
        next_date = _get_next_trading_day(date_str, daily_df)
        if next_date is None:
            continue

        # 计算次日各子方向收益和量比
        next_returns = _compute_sub_daily_return(daily_df, next_date)
        next_vol_ratios = _compute_sub_volume_ratio(daily_df, next_date)

        if not next_returns:
            continue

        for rec in records:
            sub = rec['sub_sector']
            next_ret = next_returns.get(sub)
            next_vol = next_vol_ratios.get(sub, 1.0)

            if next_ret is None:
                continue

            if sub not in stats:
                stats[sub] = {
                    'confirm_hits': 0, 'confirm_total': 0,
                    'fail_hits': 0, 'fail_total': 0,
                }

            # 确认条件命中判断：次日收益 > 0 且量比 > 1.05
            has_confirm = rec.get('confirm_condition', '').strip()
            if has_confirm:
                stats[sub]['confirm_total'] += 1
                if next_ret > BACKTEST_RETURN_THRESHOLD and next_vol > BACKTEST_VOL_RATIO_CONFIRM:
                    stats[sub]['confirm_hits'] += 1

            # 失效条件命中判断：次日收益 <= -0.5% 或量比 < 0.95
            has_fail = rec.get('fail_condition', '').strip()
            if has_fail:
                stats[sub]['fail_total'] += 1
                if next_ret <= BACKTEST_FAIL_THRESHOLD or next_vol < 0.95:
                    stats[sub]['fail_hits'] += 1

    if not stats:
        return pd.DataFrame()

    rows = []
    for sub, s in stats.items():
        total_checks = s['confirm_total'] + s['fail_total']
        confirm_hit_rate = s['confirm_hits'] / s['confirm_total'] if s['confirm_total'] > 0 else 0.0
        fail_hit_rate = s['fail_hits'] / s['fail_total'] if s['fail_total'] > 0 else 0.0
        rows.append({
            'sub_sector': sub,
            'confirm_hit_rate': round(confirm_hit_rate, 4),
            'fail_hit_rate': round(fail_hit_rate, 4),
            'confirm_hits': s['confirm_hits'],
            'confirm_total': s['confirm_total'],
            'fail_hits': s['fail_hits'],
            'fail_total': s['fail_total'],
            'total_checks': total_checks,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values('total_checks', ascending=False).reset_index(drop=True)
    logger.info(f"回测完成: {len(result)} 个子方向, "
                f"确认平均命中率: {result['confirm_hit_rate'].mean():.2%}, "
                f"失效平均命中率: {result['fail_hit_rate'].mean():.2%}")
    return result


def get_reusable_patterns(min_hit_rate=0.6, min_samples=3):
    """从历史中筛选高命中率的可复用模式。

    返回: list of dict
    """
    backtest_df = backtest_conditions()

    if backtest_df.empty:
        logger.info("回测结果为空，无可用模式")
        return []

    # 筛选：确认命中率 >= min_hit_rate 且检查次数 >= min_samples
    mask = (backtest_df['confirm_hit_rate'] >= min_hit_rate) & \
           (backtest_df['total_checks'] >= min_samples)
    candidates = backtest_df[mask].copy()

    if candidates.empty:
        logger.info(f"无满足条件的高命中率模式 (min_hit_rate={min_hit_rate}, min_samples={min_samples})")
        return []

    # 加载原始模式数据获取完整信息
    all_patterns = load_patterns(days=999)
    all_records = []
    for date_records in all_patterns.values():
        all_records.extend(date_records)

    # 按子方向分组，取最新记录
    pattern_by_sub = {}
    for rec in all_records:
        sub = rec['sub_sector']
        if sub not in pattern_by_sub or rec.get('date', '') > pattern_by_sub[sub].get('date', ''):
            pattern_by_sub[sub] = rec

    result = []
    for _, row in candidates.iterrows():
        sub = row['sub_sector']
        pattern = pattern_by_sub.get(sub, {})
        result.append({
            'sub_sector': sub,
            'shenwan': pattern.get('shenwan', ''),
            'tier': pattern.get('tier', ''),
            'confirm_hit_rate': row['confirm_hit_rate'],
            'fail_hit_rate': row['fail_hit_rate'],
            'total_checks': int(row['total_checks']),
            'confirm_condition': pattern.get('confirm_condition', ''),
            'fail_condition': pattern.get('fail_condition', ''),
            'catalyst': pattern.get('catalyst', ''),
        })

    result.sort(key=lambda x: x['confirm_hit_rate'], reverse=True)
    logger.info(f"可复用模式: {len(result)} 个")
    return result


# === 自检 ===
if __name__ == '__main__':
    print("=" * 60)
    print("fermentation_patterns 自检")
    print("=" * 60)

    # 1. 测试 save_patterns / load_patterns
    print("\n[1] 测试 save/load patterns...")
    import pandas as pd

    # 构造模拟数据
    mock_ferm_df = pd.DataFrame([
        {'sub_sector': '小金属', 'shenwan': '有色金属', 'tier': 'B',
         'position': 0.65, 'volume_ratio': 1.5, 'return_20d': 0.03,
         'breadth_3d': 0.67, 'breadth_momentum': 0.15, 'consecutive_up': 2,
         'crowding_score': 0.35, 'ferment_score': 0.62, 'n_stocks': 12},
        {'sub_sector': '白酒', 'shenwan': '食品饮料', 'tier': 'A',
         'position': 0.55, 'volume_ratio': 1.8, 'return_20d': -0.02,
         'breadth_3d': 0.75, 'breadth_momentum': 0.25, 'consecutive_up': 1,
         'crowding_score': 0.25, 'ferment_score': 0.78, 'n_stocks': 20},
        {'sub_sector': '证券', 'shenwan': '非银金融', 'tier': 'C',
         'position': 0.72, 'volume_ratio': 1.1, 'return_20d': 0.05,
         'breadth_3d': 0.50, 'breadth_momentum': -0.05, 'consecutive_up': 3,
         'crowding_score': 0.45, 'ferment_score': 0.48, 'n_stocks': 15},
    ])

    mock_llm = {
        '小金属': {'catalyst': '稀土价格反弹', 'confirm_condition': '板块放量突破5日线',
                   'fail_condition': '缩量跌破20日线'},
        '白酒': {'catalyst': '春节备货行情', 'confirm_condition': '茅台站稳1800',
                 'fail_condition': '行业库存数据恶化'},
        '证券': {'catalyst': '', 'confirm_condition': '', 'fail_condition': ''},
    }

    test_date = '2026-06-19'
    save_patterns(mock_ferm_df, mock_llm, test_date)
    print(f"  ✅ 已保存 {test_date} 的 {len(mock_ferm_df)} 条记录")

    loaded = load_patterns(days=30)
    print(f"  ✅ 加载 {len(loaded)} 天数据, 共 {sum(len(v) for v in loaded.values())} 条记录")
    assert test_date in loaded, f"缺少 {test_date} 的数据"
    assert len(loaded[test_date]) == 3, f"期望 3 条, 实际 {len(loaded[test_date])}"
    print(f"  ✅ 数据验证通过: {loaded[test_date][0]['sub_sector']} ({loaded[test_date][0]['tier']}档)")

    # 2. 测试 backtest_conditions (无实际数据时会返回空)
    print("\n[2] 测试 backtest_conditions...")
    bt_df = backtest_conditions(days=30)
    if bt_df.empty:
        print("  ⚠️ 无日线缓存数据，回测返回空 (预期行为)")
        print("  ℹ️ 提示: 运行主流水线后可获得回测结果")
    else:
        print(f"  ✅ 回测结果: {len(bt_df)} 个子方向")
        print(bt_df.to_string())

    # 3. 测试 get_reusable_patterns
    print("\n[3] 测试 get_reusable_patterns...")
    reusable = get_reusable_patterns(min_hit_rate=0.5, min_samples=1)
    if reusable:
        print(f"  ✅ 可复用模式: {len(reusable)} 个")
        for p in reusable[:3]:
            print(f"    {p['sub_sector']}: 确认命中率={p['confirm_hit_rate']:.0%}, "
                  f"检查次数={p['total_checks']}")
    else:
        print("  ⚠️ 无可复用模式 (数据不足)")

    # 4. 测试边缘情况
    print("\n[4] 边缘情况测试...")

    # 空 DataFrame
    empty_df = pd.DataFrame()
    save_patterns(empty_df, {}, '2026-06-20')
    loaded_empty = load_patterns(days=1)
    assert '2026-06-20' in loaded_empty
    assert len(loaded_empty['2026-06-20']) == 0
    print("  ✅ 空 DataFrame 处理正确")

    # 无 LLM 增强
    no_llm_df = pd.DataFrame([{
        'sub_sector': '国防军工', 'shenwan': '国防军工', 'tier': 'B',
        'position': 0.60, 'volume_ratio': 1.3, 'return_20d': 0.01,
        'breadth_3d': 0.55, 'breadth_momentum': 0.10, 'consecutive_up': 2,
        'crowding_score': 0.30, 'ferment_score': 0.60, 'n_stocks': 10,
    }])
    save_patterns(no_llm_df, {}, '2026-06-21')
    assert load_patterns(days=1)['2026-06-21'][0]['catalyst'] == ''
    print("  ✅ 无 LLM 增强处理正确")

    # 文件不存在
    if os.path.exists(PATTERNS_FILE):
        # 备份后删除测试
        backup = PATTERNS_FILE + '.bak'
        os.rename(PATTERNS_FILE, backup)
        empty_load = load_patterns()
        assert empty_load == {}
        print(f"  ✅ 文件不存在返回空字典: {len(empty_load)} 条")
        os.rename(backup, PATTERNS_FILE)

    print("\n" + "=" * 60)
    print("✅ 所有自检通过")
    print("=" * 60)
