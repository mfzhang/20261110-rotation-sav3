#!/usr/bin/env python3
# /tmp/sav3/main.py
# 投资建议系统 — ARC (Adaptive Rotation Core)
# 多因子选股 + 行业轮动 + 持仓分析 + 危机检测
import os, sys, logging, json, re, threading, time, urllib.request
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import *
from data_preparation import prepare_data
from factor_calculator import calculate_factors
from factor_pipeline import pipeline_process
from score_ranker import compute_composite_score, rank_stocks, compute_rolling_ic, compute_regime_weights
from crisis_detector import CrisisDetector
import data_fetcher
from pipeline_types import FactorPipelineResult
from report_generator import ReportGenerator
import sector_rotation
from market_timing import MarketTiming as PSMTiming
try:
    from fermentation import run_fermentation_pipeline
    FERMENTATION_AVAILABLE = True
except ImportError:
    FERMENTATION_AVAILABLE = False
from sub_sectors import get_all_sub_sectors
import portfolio

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("main")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PREV_RANK_FILE = os.path.join(OUTPUT_DIR, 'prev_top20.json')

PSM_LINKAGE_ENABLED = True  # 全局开关：PSM Q3 展示级概率修正（不联动交易）

CACHE_MAX_AGE_DAYS = 30  # 行情缓存（ts_daily_*/ts_idx_*）最大保留天数


def _cleanup_old_cache(max_age_days: int = CACHE_MAX_AGE_DAYS):
    """启动时清理超过 max_age_days 未更新的行情缓存文件。

    只针对 ts_daily_* / ts_idx_*（含 .meta.json）两类行情缓存；
    data_fetcher._cleanup_stale_cache 依赖成功写缓存才触发，本函数在每次
    运行启动时无条件兜底，防止旧命名/废弃哈希缓存无限堆积。
    """
    try:
        cache_dir = data_fetcher.CACHE_DIR
        cutoff = time.time() - max_age_days * 86400
        removed = 0
        for f in os.listdir(cache_dir):
            if not (f.startswith('ts_daily_') or f.startswith('ts_idx_')):
                continue
            path = os.path.join(cache_dir, f)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
                    logger.info(f"清理过期缓存: {f}")
            except OSError:
                pass
        if removed:
            logger.info(f"缓存清理完成: 删除 {removed} 个超过 {max_age_days} 天的缓存文件")
    except Exception as e:
        logger.warning(f"缓存清理失败（不影响主流程）: {e}")


def _load_prev_top20() -> dict:
    """加载昨日 Top20 排名，用于计算排名变化。"""
    if os.path.exists(PREV_RANK_FILE):
        try:
            with open(PREV_RANK_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _append_news_sent_history(news_sent, target_date: str):
    """落盘每日行业情绪（append-only JSONL，同日重跑去重）。

    用途: 新闻 δ 的真实回测需要历史 sentiment；tushare news 无权限、爬虫源只有当日，
    从这里开始积累。文件: output/news_sent_history.jsonl，每行一次运行。
    """
    path = os.path.join(OUTPUT_DIR, 'news_sent_history.jsonl')
    df = news_sent.copy()
    df['date'] = df['date'].astype(str)
    record = json.dumps({'run_date': target_date, 'records': df.to_dict('records')},
                        ensure_ascii=False)
    lines = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l for l in f
                     if l.strip() and f'"run_date": "{target_date}"' not in l]
    with open(path, 'w', encoding='utf-8') as f:
        for l in lines:
            f.write(l if l.endswith('\n') else l + '\n')
        f.write(record + '\n')


def _save_prev_top20(ranked_df, n=20):
    """保存当前 TopN 排名供次日对比。"""
    codes = ranked_df.head(n).index.tolist()
    with open(PREV_RANK_FILE, 'w') as f:
        json.dump(codes, f)


def _compute_arc_health(ranked, daily_df, industry_map) -> dict:
    """计算 ARC 模型健康指标。"""
    health = {}

    # 行业集中度 HHI（Top50 中每个行业占比的平方和）
    if ranked is not None and not ranked.empty and 'industry' in ranked.columns:
        top50 = ranked.head(50)
        ind_pct = top50['industry'].value_counts(normalize=True)
        health['hhi'] = (ind_pct ** 2).sum()
        health['max_industry'] = ind_pct.index[0]
        health['max_industry_pct'] = ind_pct.iloc[0]
    else:
        health['hhi'] = None

    # 最近5日因子IC（如果有足够数据）
    health['ic_5d'] = None

    return health


def _compute_system_health(result, mt_result, target_date: str) -> dict:
    """采集系统健康指标（日报"系统健康"区）。

    本次 4 个 P0 的共性是静默损坏——此面板是总防线：
    各数据源行数/最新日期、因子 NaN 率（>5% 红旗）、PSM 指数完整性、
    危机状态溯源、缓存规模。任何一项异常都应在日报上可见。
    """
    h = {'red_flags': []}
    target_dt = pd.to_datetime(target_date)

    # ── 个股日K ──
    daily = result.daily_df
    if daily is not None and not daily.empty:
        dates = daily.index.get_level_values('date')
        latest = dates.max()
        stale = (target_dt - latest).days
        h['stock_daily'] = {
            'rows': len(daily),
            'codes': int(daily.index.get_level_values('code').nunique()),
            'latest': latest.strftime('%Y-%m-%d'),
        }
        if stale > 5:
            h['red_flags'].append(f"个股日K陈旧 {stale} 天（最新 {latest.date()}）")

    # ── 因子 NaN 率（P0-1 防复发：任何因子 NaN 率 >5% 亮红灯）──
    factors = result.factors_df
    if factors is not None and not factors.empty:
        latest_f = factors.index.get_level_values('date').max()
        cross = factors.xs(latest_f, level='date')
        nan_rates = {}
        for fac in ['low_risk', 'position']:
            if fac in cross.columns:
                rate = float(cross[fac].isna().mean())
                nan_rates[fac] = rate
                if rate > 0.05:
                    h['red_flags'].append(f"因子 {fac} NaN 率 {rate:.0%}（>5% 阈值）")
        h['factor_nan'] = nan_rates
        h['factor_latest'] = latest_f.strftime('%Y-%m-%d')
    else:
        h['red_flags'].append("因子面板为空")

    # ── 基准指数（沪深300）──
    idx_df = result.index_df
    if idx_df is not None and not idx_df.empty:
        latest_i = idx_df.index.max()
        h['index_hs300'] = {'rows': len(idx_df), 'latest': latest_i.strftime('%Y-%m-%d')}
        if (target_dt - latest_i).days > 5:
            h['red_flags'].append(f"沪深300指数陈旧（最新 {latest_i.date()}）")

    # ── PSM 六指数完整性（P0-3 防线）──
    if mt_result and mt_result.get('data_health'):
        dh = mt_result['data_health']
        h['psm_indices'] = dh
        missing = mt_result.get('data_missing', [])
        if missing:
            h['red_flags'].append(f"PSM 指数缺失: {', '.join(missing)}")
        else:
            for name, info in dh.items():
                if info.get('latest'):
                    stale = (target_dt - pd.Timestamp(info['latest'])).days
                    if stale > 7:
                        h['red_flags'].append(f"PSM {name} 陈旧 {stale} 天")

    # ── 危机状态溯源（P0-2 防线）──
    det = result.detector
    if det is not None:
        h['crisis'] = {
            'mode': det.mode,
            'entry_date': str(det.entry_date.date()) if getattr(det, 'entry_date', None) else None,
            'trigger_reason': getattr(det, 'trigger_reason', None),
        }
        if det.mode == 'crisis' and not getattr(det, 'entry_date', None):
            h['red_flags'].append("危机状态来源不明（crisis 无 entry_date，疑似历史污染残留）")

    # ── 缓存规模 ──
    try:
        import data_fetcher as _df_mod
        cache_dir = _df_mod.CACHE_DIR
        total_mb = sum(os.path.getsize(os.path.join(cache_dir, f))
                       for f in os.listdir(cache_dir) if f.endswith('.pkl')) / 1e6
        h['cache_mb'] = round(total_mb, 0)
        if total_mb > 2000:
            h['red_flags'].append(f"缓存目录 {total_mb:.0f}MB 超 2GB")
    except Exception:
        pass

    return h


def _adjust_psm_for_display(mt_result, health, ranked, ferment_result):
    """
    Q3: ARC/TID → PSM 展示级概率修正。
    只影响日报展示，不反馈回 ARC/TID 的交易决策。
    """
    if mt_result is None or not mt_result.get('scenarios'):
        return None

    scenarios = mt_result['scenarios']
    p1 = scenarios.get('scenario_1', {}).get('probability', 0)
    p2 = scenarios.get('scenario_2', {}).get('probability', 0)
    p3 = scenarios.get('scenario_3', {}).get('probability', 0)

    # ARC 信号
    arc_hhi = health.get('hhi') if health else None
    arc_avg_position = None
    # 用 [0,1] 的 position_rank（与 0.30/0.70 阈值设计口径一致）；
    # 原先用中性化后的 z 分数，阈值语义错位且被格式化成百分比展示（review §3.3）
    if ranked is not None and not ranked.empty:
        if 'position_rank' in ranked.columns:
            arc_avg_position = ranked.head(20)['position_rank'].mean()
        elif 'position' in ranked.columns:
            arc_avg_position = ranked.head(20)['position'].mean()

    # TID 信号
    total_sub_sectors = max(len(get_all_sub_sectors()), 1)
    tid_breadth = None
    tid_dispersion = None
    crowding = ferment_result.get('crowding') if ferment_result else None
    fermentation_df = ferment_result.get('fermentation') if ferment_result else None
    if fermentation_df is not None and not fermentation_df.empty:
        tid_breadth = len(fermentation_df) / total_sub_sectors
    if crowding is not None and not crowding.empty and 'return_20d' in crowding.columns:
        tid_dispersion = crowding['return_20d'].max() - crowding['return_20d'].min()

    tid_crowded_count = len(ferment_result.get('excluded_crowded', [])) if ferment_result else 0

    # 以原始概率为起点做修正
    adj_p1, adj_p2, adj_p3 = p1, p2, p3
    if arc_hhi is not None and arc_hhi > 0.15:
        adj_p1 -= 0.03
        adj_p3 += 0.03
    if arc_avg_position is not None and arc_avg_position < 0.30:
        adj_p3 += 0.04
    if tid_breadth is not None and tid_breadth < 0.40:
        adj_p3 += 0.03
    if total_sub_sectors > 0 and (tid_crowded_count / total_sub_sectors) > 0.20:
        adj_p3 += 0.02
    if tid_dispersion is not None and tid_dispersion > 0.15:
        adj_p3 += 0.02

    # 对称偏暖信号
    if arc_avg_position is not None and arc_avg_position > 0.70:
        adj_p1 += 0.03
        adj_p3 -= 0.03
    if tid_breadth is not None and tid_breadth > 0.60:
        adj_p1 += 0.02

    # 约束：每个概率 ∈ [0.10, 0.60]，并归一化和为 1
    adj_p1 = max(0.10, min(0.60, adj_p1))
    adj_p2 = max(0.10, min(0.60, adj_p2))
    adj_p3 = max(0.10, min(0.60, adj_p3))

    total = adj_p1 + adj_p2 + adj_p3
    if total > 0:
        adj_p1 /= total
        adj_p2 /= total
        adj_p3 /= total
    else:
        adj_p1, adj_p2, adj_p3 = p1, p2, p3

    return {
        'original': {'scenario_1': p1, 'scenario_2': p2, 'scenario_3': p3},
        'adjusted': {'scenario_1': adj_p1, 'scenario_2': adj_p2, 'scenario_3': adj_p3},
        'signals': {
            'arc_hhi': arc_hhi,
            'arc_avg_position': arc_avg_position,
            'tid_breadth': tid_breadth,
            'tid_crowded_count': tid_crowded_count,
            'tid_dispersion': tid_dispersion,
        }
    }


def _compute_psm_cross_validation(mt_result, psm_adjusted=None) -> str:
    """
    Q4: PSM → TID 展示级交叉验证（仅展示，不动交易）。

    根据 PSM 三剧本概率判断当前市场水温，返回统一标注:
      ✅  bullish: PSM 看涨，低位可能是买入机会
      ⚠️  bearish: PSM 看跌，低位可能是陷阱
      ➖  neutral: 无明显偏向

    阈值与 Q3 保持一致；优先使用 Q3 修正后的概率。
    """
    if mt_result is None or not mt_result.get('scenarios'):
        return '➖'

    # 优先使用 Q3 修正后概率
    if psm_adjusted is not None and 'adjusted' in psm_adjusted:
        adj = psm_adjusted['adjusted']
        bull = adj.get('scenario_1', 0) + adj.get('scenario_2', 0)
        bear = adj.get('scenario_3', 0)
    else:
        scenarios = mt_result['scenarios']
        bull = scenarios.get('scenario_1', {}).get('probability', 0) + \
               scenarios.get('scenario_2', {}).get('probability', 0)
        bear = scenarios.get('scenario_3', {}).get('probability', 0)

    if bull > 0.60:
        return '✅'
    elif bear > 0.40:
        return '⚠️'
    else:
        return '➖'


def run_factor_pipeline(target_date: str, stock_limit: int = 300, skip_financials: bool = True) -> tuple:
    """运行 ARC 因子流水线，返回 (ranked_df, FactorPipelineResult)"""

    target_dt = pd.to_datetime(target_date)
    start_data = (target_dt - pd.DateOffset(years=2)).strftime('%Y-%m-%d')

    # 1. 数据准备
    logger.info(f"数据准备: {start_data} ~ {target_date} (limit={stock_limit})")
    daily, idx_ret, ind, index_df = prepare_data(start_data, target_date,
                                                  stock_limit=stock_limit, skip_financials=skip_financials)
    logger.info(f"数据: {len(daily):,} 条日K记录")

    # 2. 因子计算
    logger.info("计算因子...")
    factors = calculate_factors(daily, idx_ret)
    # 保留原始因子面板（pipeline_process 会中性化+标准化成 z 分数，
    # 破坏 position 的 [0,1] 语义；TID×ARC 交叉验证需要原始口径）
    raw_factors = factors

    # 3. 因子处理（中性化+标准化）
    logger.info("因子处理...")
    factors = pipeline_process(factors, ind)

    # 4. 危机检测（多指数加权：沪深300/创业板指/科创50，config 配置正式接线）
    index_close = index_df['close']
    index_closes = {'000300.SH': index_close}  # 兜底：单指数模式
    try:
        from index_data import INDEX_CODES
        all_idx = data_fetcher.fetch_indices_daily(start_data, target_date, INDEX_CODES,
                                                   fields=['close'])
        code_to_name = {v: k for k, v in INDEX_CODES.items()}
        multi = {}
        for code in CRISIS_INDEX_CODES:
            name = code_to_name.get(code)
            if name and name in all_idx and not all_idx[name].empty:
                multi[code] = all_idx[name]['close']
        if '000300.SH' in multi and len(multi) >= 2:
            index_closes = multi
            logger.info(f"危机检测: 多指数加权模式 ({len(multi)} 指数)")
        else:
            logger.warning("危机检测: 多指数数据不全，降级单指数(沪深300)模式")
    except Exception as e:
        logger.warning(f"危机检测多指数拉取失败，降级单指数模式: {e}")

    detector = CrisisDetector(index_drawdown=CRISIS_IDX_DRAWDOWN, vol_ratio=CRISIS_VOL_RATIO,
                              exit_drawdown=CRISIS_EXIT_DRAWDOWN, exit_vol_ratio=CRISIS_EXIT_VOL_RATIO,
                              exit_days=CRISIS_EXIT_DAYS,
                              cooling_days=CRISIS_COOLING_DAYS,
                              stock_stoploss=CRISIS_STOCK_STOPLOSS,
                              index_codes=CRISIS_INDEX_CODES,
                              index_weights=CRISIS_INDEX_WEIGHTS)
    crisis_mode, need_rebalance = detector.check(index_closes, target_dt)
    for w in detector.warnings:
        logger.warning(f"危机检测警告: {w}")
    logger.info(f"危机检测: mode={crisis_mode}, need_rebalance={need_rebalance}")

    # 5. 加权排名（P1-6: 按市场状态动态配权——position 是崩盘期权，
    #    崩盘/超跌加权 position，趋势市降权，详见 review §3.1）
    weights, regime = compute_regime_weights(index_closes.get('000300.SH', index_close),
                                             crisis_mode)
    logger.info(f"市场状态: {regime}, 因子权重: {weights}")

    available_dates = factors.index.get_level_values('date').unique()
    if target_dt not in available_dates:
        target_dt = available_dates[-1]
        logger.info(f"目标日期非交易日，使用最近交易日: {target_dt.date()}")

    scores = compute_composite_score(factors, weights=weights)

    # 涨跌停当日不可交易：从当日候选剔除 tradable=False
    # （P1-5: data_preparation 改为打标记不删行，价格序列保持连续）
    try:
        tradable_today = daily.xs(target_dt, level='date')['tradable']
        blocked = set(tradable_today[~tradable_today].index)
        if blocked:
            mask = ~((scores.index.get_level_values('date') == target_dt) &
                     scores.index.get_level_values('code').isin(list(blocked)))
            scores = scores[mask]
            logger.info(f"涨跌停过滤: 当日 {len(blocked)} 只不可交易已从候选剔除")
    except KeyError:
        pass

    result = rank_stocks(scores, date=target_dt)

    # P0#1: 危机模式下按 CRISIS_EXPOSURE 缩放 composite_score，体现 40% 仓位
    if crisis_mode == 'crisis':
        result['composite_score'] *= CRISIS_EXPOSURE
        logger.info(f"危机模式：composite_score 已缩放至 {CRISIS_EXPOSURE:.0%}")

    # 6. 获取收盘价（用于买入参考价）
    try:
        close_today = daily.xs(target_dt, level='date')['close']
        result['close'] = result.index.map(close_today)
    except Exception:
        result['close'] = None

    # 7. 整理信号
    basic = data_fetcher.get_stock_basic()
    name_col = 'name' if 'name' in basic.columns else 'code_name'
    code_col = 'ts_code' if 'ts_code' in basic.columns else 'code'
    name_map = basic.set_index(code_col)[name_col]
    result['name'] = result.index.map(name_map)
    result['industry'] = result.index.map(ind)

    pipeline_result = FactorPipelineResult(
        crisis_mode=crisis_mode,
        need_rebalance=need_rebalance,
        weights_used=weights,
        index_latest=float(index_close.iloc[-1]) if len(index_close) > 0 else None,
        daily_df=daily,
        industry_map=ind,
        detector=detector,  # P1#3: 供 generate_report 调用 check_stock_stoploss
        factors_df=factors,  # 因子截面数据，供 IC 计算
        raw_factors_df=raw_factors,  # 原始因子（position [0,1]），供 TID×ARC 交叉验证
        regime=regime,
        index_df=index_df,
    )

    return result, pipeline_result


def _generate_ferm_llm_enhance(ferment_result: dict, news_detailed: dict, psm_context: str = ""):
    """用 DeepSeek API 为 A/B 档生成催化判断 + 确认条件 + 失效条件"""
    ferm_df = ferment_result['fermentation']
    ab_mask = ferm_df['tier'].isin(['A', 'B'])
    if not ab_mask.any():
        return

    ab_df = ferm_df[ab_mask]
    headlines_by_sub = {}
    if news_detailed and news_detailed.get('headlines'):
        for h in news_detailed['headlines']:
            for sub in h.get('sub_sectors', []):
                if sub not in headlines_by_sub:
                    headlines_by_sub[sub] = []
                headlines_by_sub[sub].append(h)

    # 构建 prompt
    prompt_lines = []
    prompt_lines.append("你是A股行业分析助手。以下是有低位发酵信号的子方向，请为每个方向生成：")
    prompt_lines.append("1. 催化判断（一句话）")
    prompt_lines.append("2. 明日确认条件（什么情况算确认）")
    prompt_lines.append("3. 失效条件（什么情况降级）\n")

    if psm_context:
        prompt_lines.append(f"当前大盘环境（PSM 概率剧本）: {psm_context}\n")
        prompt_lines.append("请让确认条件/失效条件适度参考上述大盘环境：若偏空，确认条件需更严格；若偏多，失效条件需更敏感。\n")

    for _, row in ab_df.iterrows():
        sub = row['sub_sector']
        news = headlines_by_sub.get(sub, [])
        news_summary = "; ".join([n['title'][:60] for n in news[:5]]) if news else "无相关新闻"
        prompt_lines.append(
            f"【{sub}】{row['shenwan']} | 52周位置:{row['position']:.0%} | "
            f"量比:{row['volume_ratio']:.1f}x | 20日涨幅:{row['return_20d']:+.1%} | "
            f"分档:{row['tier']}\n新闻: {news_summary}"
        )

    prompt_lines.append("\n输出纯JSON数组，每个元素: {sub_sector, catalyst, confirm_condition, fail_condition}")
    prompt = "\n".join(prompt_lines)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        for candidate in [os.path.expanduser('~/.hermes/profiles/sa-system/.env'),
                         '/root/.hermes/profiles/sa-system/.env']:
            if os.path.exists(candidate):
                with open(candidate) as f:
                    for line in f:
                        if line.startswith('DEEPSEEK_API_KEY='):
                            api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                            break
                break
    if not api_key:
        logger.warning("无 DEEPSEEK_API_KEY，跳过 LLM 增强")
        return

    try:
        payload = json.dumps({
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是A股行业分析助手。输出纯JSON数组，无解释文字。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,  # 与 news_scraper 情感分类一致，确定性输出（可复现）
            "max_tokens": 2048
        }).encode()
        req = urllib.request.Request(
            "https://api.deepseek.com/v1/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            result = json.loads(resp.read())
        content = result["choices"][0]["message"]["content"]
        # 解析 JSON
        cleaned = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        llm_data = json.loads(cleaned)
        if isinstance(llm_data, list):
            ferm_subs = set(ferment_result['fermentation']['sub_sector'].values)
            for item in llm_data:
                raw_sub = item.get('sub_sector', '').strip()
                if raw_sub in ferm_subs:
                    ferment_result.setdefault('llm_enhance', {})[raw_sub] = item
                else:
                    for fsub in ferm_subs:
                        if raw_sub in fsub or fsub in raw_sub:
                            ferment_result.setdefault('llm_enhance', {})[fsub] = item
                            break
            logger.info(f"LLM 增强: {len(llm_data)} 个子方向")
            # 保存模式数据（供后续回测确认/失效条件命中率）
            try:
                from fermentation_patterns import save_patterns
                save_patterns(ferment_result['fermentation'],
                             ferment_result.get('llm_enhance', {}), target_date)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"LLM 增强失败: {e}")


def generate_report(target_date: str = None, stock_limit: int = 300, skip_financials: bool = True):
    """生成完整投资建议报告"""
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')

    _cleanup_old_cache()  # 启动时清理超过 CACHE_MAX_AGE_DAYS 天的行情缓存

    report_path = os.path.join(OUTPUT_DIR, 'daily_advice.md')
    weekday = ['一', '二', '三', '四', '五', '六', '日'][datetime.strptime(target_date, '%Y-%m-%d').weekday()]

    # ===== 1. ARC 多因子选股 =====
    logger.info("=== ARC 多因子选股 ===")
    ranked, result = None, FactorPipelineResult()
    try:
        ranked, result = run_factor_pipeline(target_date, stock_limit=stock_limit,
                                             skip_financials=skip_financials)
    except Exception as e:
        logger.error(f"因子选股失败: {e}", exc_info=True)

    # ===== 2. 新闻抓取（一次 API 调用，供 TID 板块轮动 + 低位发酵共用）=====
    logger.info("=== 新闻抓取 ===")
    daily_df = result.daily_df
    ind_map = result.industry_map
    news_detailed = None
    news_sent = None
    try:
        from news_scraper import scrape_daily_news_detailed
        news_detailed = scrape_daily_news_detailed(target_date)
        if news_detailed and news_detailed.get('headlines'):
            logger.info(f"详细新闻: {news_detailed['total_count']}条头条")
            # 从 detailed 结果提取行业聚合 DataFrame，供 TID 板块轮动使用
            news_sent = news_detailed['aggregated']
    except Exception as e:
        logger.error(f"新闻抓取失败: {e}")

    # δ 真实回测数据积累：落盘当日行业情绪（append-only，同日重跑去重）
    try:
        if news_sent is not None and not news_sent.empty:
            _append_news_sent_history(news_sent, target_date)
    except Exception as e:
        logger.warning(f"新闻情绪落盘失败（不影响报告）: {e}")

    # ===== 2a. TID 板块轮动 =====
    logger.info("=== TID 板块轮动 ===")
    top3_raw, bottom3_raw, perf, tid_picks = [], [], None, []
    try:
        # use_full_market=True: 行业指数用全市场口径（修复 300 股随机池抽样偏差）；
        # 全市场拉取失败时 sector_rotation 内部自动降级为共享池口径
        top3_raw, bottom3_raw, perf = sector_rotation.recommend(
            daily_df=daily_df, industry_map=ind_map, events_sent=news_sent,
            use_full_market=True)
        tid_picks = sector_rotation.get_stock_picks(
            daily_df=daily_df, industry_map=ind_map,
            top_n_industries=3, picks_per_industry=5,
            events_sent=news_sent, use_full_market=True)
    except Exception as e:
        logger.error(f"板块轮动失败: {e}")

    # ===== 2b. PSM 概率剧本 =====
    logger.info("=== PSM 概率剧本 ===")
    mt_result = None
    try:
        mt = PSMTiming(end_date=target_date)
        mt_result = mt.run_full_analysis()
        if mt_result:
            logger.info(f"PSM: 3剧本已生成, "
                       f"严格口径{mt_result.get('backtest',{}).get('strict',{}).get('strict_matches',0)}次, "
                       f"宽口径{mt_result.get('backtest',{}).get('wide',{}).get('wide_matches',0)}次")
    except Exception as e:
        logger.error(f"PSM 失败: {e}")

    # ===== 2c. 低位发酵检测 =====
    ferment_result = None
    if FERMENTATION_AVAILABLE and daily_df is not None and ind_map is not None:
        logger.info("=== 低位发酵检测 ===")
        try:
            ferment_result = run_fermentation_pipeline(daily_df, ind_map)
            if ferment_result and not ferment_result['fermentation'].empty:
                logger.info(f"发酵候选: A={len(ferment_result['fermentation'][ferment_result['fermentation']['tier']=='A'])}, "
                           f"B={len(ferment_result['fermentation'][ferment_result['fermentation']['tier']=='B'])}, "
                           f"C={len(ferment_result['fermentation'][ferment_result['fermentation']['tier']=='C'])}")
        except Exception as e:
            logger.error(f"低位发酵检测失败: {e}")

        # LLM 增强 + 交叉验证（复用已抓取的 news_detailed）
        if news_detailed and news_detailed.get('headlines'):
            try:
                if ferment_result and not ferment_result['fermentation'].empty:
                    # 构造 PSM 市场状态供 LLM 参考
                    psm_context = ""
                    if PSM_LINKAGE_ENABLED and mt_result and mt_result.get('scenarios'):
                        s = mt_result['scenarios']
                        if s.get('probabilities_available'):
                            p1 = s.get('scenario_1', {}).get('probability') or 0
                            p2 = s.get('scenario_2', {}).get('probability') or 0
                            p3 = s.get('scenario_3', {}).get('probability') or 0
                            bull_score = p1 + p2
                            regime = "偏空" if p3 > 0.40 else ("偏多" if bull_score > 0.60 else "震荡")
                            psm_context = f"逼空概率 {bull_score:.0%}，兑现失败概率 {p3:.0%}，整体判断：{regime}"
                        else:
                            # 仪表盘模式：概率抑制，改用加权20日涨跌幅定性描述
                            dash = s.get('dashboard', {})
                            wr = dash.get('weighted_ret_20d')
                            n_wide = dash.get('sample', {}).get('n_wide', 0)
                            if wr is not None:
                                regime = "偏空" if wr < -0.05 else ("偏多" if wr > 0.05 else "震荡")
                                psm_context = (f"PSM 样本不足(n={n_wide}<20)不输出概率；"
                                               f"加权20日涨跌幅 {wr:+.1%}，整体判断：{regime}")
                    _generate_ferm_llm_enhance(ferment_result, news_detailed, psm_context=psm_context)
                    # TID × ARC 交叉验证（复用 factor_calculator 原始 position 输出）
                    try:
                        from fermentation_cross import run_cross_validation
                        if daily_df is not None and ind_map is not None:
                            ferment_result['fermentation'] = run_cross_validation(
                                daily_df, ind_map,
                                ferment_result['fermentation'],
                                factors_df=result.raw_factors_df)
                            arc_confirmed = ferment_result['fermentation']['arc_confirm'].sum()
                            logger.info(f"ARC 交叉验证: {int(arc_confirmed)} 个子方向通过")
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"LLM增强/交叉验证失败: {e}")

    # ===== 2d. A/B/C 分档命中率分析（供日报发酵区引用，并落盘 latest.json） =====
    ferm_hitrate = None
    try:
        from ferm_hitrate_analyzer import analyze_tier_hit_rates, save_latest
        ferm_hitrate = analyze_tier_hit_rates(days=20, daily_df=daily_df)
        if ferm_hitrate:
            save_latest(ferm_hitrate)
    except Exception as e:
        logger.warning(f"分档命中率分析失败（不影响报告）: {e}")

    # 模型健康（供 Q3 使用）
    crisis_mode = result.crisis_mode
    weights_used = result.weights_used
    health = _compute_arc_health(ranked, result.daily_df, result.industry_map)

    # 系统健康（日报防线：数据源/因子NaN率/PSM完整性/危机溯源/缓存）
    system_health = _compute_system_health(result, mt_result, target_date)
    if system_health.get('red_flags'):
        for flag in system_health['red_flags']:
            logger.warning(f"🚩 系统健康红旗: {flag}")

    # P1: 因子 IC 监控（horizon=20 ≈ 月度调仓周期对齐，review §3.4）
    rolling_ic = None
    factors_df = result.factors_df
    daily_df_ic = result.daily_df
    if factors_df is not None and daily_df_ic is not None:
        try:
            rolling_ic = compute_rolling_ic(factors_df, daily_df_ic, n_periods=12, horizon=20)
            logger.info(f"因子IC: low_risk IC={rolling_ic.get('low_risk',{}).get('ic_mean',0):+.3f} "
                       f"IR={rolling_ic.get('low_risk',{}).get('ic_ir',0):.2f} | "
                       f"position IC={rolling_ic.get('position',{}).get('ic_mean',0):+.3f} "
                       f"IR={rolling_ic.get('position',{}).get('ic_ir',0):.2f}")
        except Exception as e:
            logger.error(f"因子IC计算失败: {e}")

    # Q3: PSM 展示级概率修正（仅展示，不反馈回交易决策）
    # 2026-07-19 仪表盘模式：PSM 样本 n<20 时不输出概率，Q3/Q4 联动同步跳过
    psm_adjusted = None
    psm_cross_mark = '➖'
    psm_probs_available = bool(
        mt_result and mt_result.get('scenarios', {}).get('probabilities_available', False))
    if PSM_LINKAGE_ENABLED and psm_probs_available:
        psm_adjusted = _adjust_psm_for_display(mt_result, health, ranked, ferment_result)
        psm_cross_mark = _compute_psm_cross_validation(mt_result, psm_adjusted)
        logger.info(f"[PSM 联动] Q3 概率修正已生成, Q4 PSM验证={psm_cross_mark}")
    elif PSM_LINKAGE_ENABLED and mt_result:
        logger.info("[PSM 联动] PSM 样本不足(n<20)不输出概率，Q3/Q4 跳过（仪表盘模式）")

    top_n = 20  # ARC 选股展示数量（不受 PSM 联动影响）

    # ===== 3. 持仓分析 =====
    logger.info("=== 持仓分析 ===")
    holdings_advice = []
    stoploss_stocks = []
    try:
        suspended_codes = set()  # 停牌股，后续跳过止损检查
        # P0: 从 daily_df 提取持仓收盘价（tushare），sina_api 降级路径已移除
        # daily_df 未覆盖的持仓股（如 ST、停牌、池外股）走 fetch_holdings_closes 小缓存
        prices_override = None
        if daily_df is not None:
            try:
                all_holdings = portfolio.load_holdings()
                prices_override = {}
                missing_codes_map = {}  # {portfolio_code: ts_code}
                target_dt = pd.to_datetime(target_date)
                for h in all_holdings:
                    portfolio_code = h['code']  # e.g. sz.000100
                    # 转换为 tushare 格式: sz.000100 → 000100.SZ, sh.600905 → 600905.SH
                    parts = portfolio_code.split('.')
                    if len(parts) == 2:
                        exchange, code_num = parts
                        ts_code = f"{code_num}.{exchange.upper()}"
                        try:
                            close_val = daily_df.loc[(target_dt, ts_code), 'close']
                            if not pd.isna(close_val):
                                prices_override[portfolio_code] = float(close_val)
                        except KeyError:
                            # 不在选股池，记录待补拉
                            missing_codes_map[portfolio_code] = ts_code

                # 按需补拉：daily_df 未覆盖的持仓股（ST/停牌/池外），走独立小缓存
                # （ts_hold_{code}.pkl 增量更新），不再每次全量 pro.daily 拉 60 天
                if missing_codes_map:
                    target_date_int = int(target_dt.strftime('%Y%m%d'))
                    hold_data = data_fetcher.fetch_holdings_closes(
                        list(missing_codes_map.values()), target_dt)
                    for pf_code, ts_code in missing_codes_map.items():
                        df = hold_data.get(ts_code)
                        if df is None or df.empty:
                            continue  # 缓存+API 均无数据，不进 prices_override
                        # 停牌股：取最后一个有效收盘价（非0）
                        valid = df[df['close'] > 0]
                        if valid.empty:
                            continue
                        prices_override[pf_code] = float(valid['close'].iloc[-1])
                        # 检测是否停牌：目标日期无交易记录 → 当日停牌
                        trade_dates = set(int(d) for d in df['trade_date'].values)
                        if target_date_int not in trade_dates:
                            suspended_codes.add(pf_code)

                if prices_override:
                    covered = len(prices_override)
                    total = len(all_holdings)
                    logger.info(f"持仓收盘价（tushare全覆盖）: {covered}/{total} 只覆盖")
            except Exception as e:
                logger.warning(f"提取持仓收盘价失败: {e}")
                prices_override = None

        holdings_advice = portfolio.analyze_holdings(prices_override=prices_override)
        # 停牌股：覆盖建议为「停牌中」，显示停牌前最后收盘价
        for h in holdings_advice:
            if h['code'] in suspended_codes:
                h['advice'] = '🛑 停牌中（无法交易）'
                h['pct_change'] = None
        # P1#3: 调用 CrisisDetector.check_stock_stoploss() 检查个股止损
        detector = result.detector
        if detector and holdings_advice:
            # 排除停牌股，避免误触止损
            active_holdings = [
                h for h in holdings_advice
                if isinstance(h.get('current'), (int, float))
                and h['code'] not in suspended_codes
                and '停牌' not in str(h.get('advice', ''))
            ]
            current_series = pd.Series({
                h['code']: h['current']
                for h in active_holdings
            })
            entry_series = pd.Series({
                h['code']: h['cost']
                for h in active_holdings
                if isinstance(h.get('cost'), (int, float)) and h['cost'] > 0
            })
            stoploss_stocks = detector.check_stock_stoploss(current_series, entry_series)
            if stoploss_stocks:
                logger.warning(f"个股止损触发: {stoploss_stocks}")
                # 更新对应持仓的建议
                for h in holdings_advice:
                    if h['code'] in stoploss_stocks:
                        h['advice'] = f"🚨 触发止损（跌幅超{abs(detector.stock_stoploss):.0%}），建议立即清仓"
    except Exception as e:
        logger.error(f"持仓分析失败: {e}")

    # ===== 5. 昨日对比 =====
    prev_top20 = _load_prev_top20()
    if ranked is not None and not ranked.empty:
        _save_prev_top20(ranked, n=top_n)

    # ===== 5b. 纸面净值追踪（Top50 等权日频，不含成本） =====
    # 每天跑完选股后把 Top50 名单落入 nav_history.json；
    # 收益口径 = 上一记录日 Top50 持有到今日的等权收益（停牌股按最后收盘价结转）
    nav_perf = None
    try:
        if ranked is not None and not ranked.empty and result.daily_df is not None:
            import nav_tracker
            top50_codes = ranked.head(TOP_N).index.tolist()
            nav_tracker.update_nav_history(top50_codes, result.daily_df)
            idx_close = result.index_df['close'] if result.index_df is not None else None
            nav_perf = nav_tracker.get_nav_performance(index_close=idx_close)
            if nav_perf:
                logger.info(f"纸面净值: {nav_tracker.format_nav_line(nav_perf)}")
    except Exception as e:
        logger.warning(f"纸面净值追踪失败（不影响报告）: {e}")

    # ===== 6. 生成报告 =====
    generator = ReportGenerator(
        target_date=target_date,
        weekday=weekday,
        signals=result,
        crisis_mode=crisis_mode,
        weights_used=weights_used,
        rolling_ic=rolling_ic,
        ranked=ranked,
        prev_top20=prev_top20,
        top_n=top_n,
        health=health,
        stoploss_stocks=stoploss_stocks,
        holdings_advice=holdings_advice,
        top3_raw=top3_raw,
        bottom3_raw=bottom3_raw,
        tid_picks=tid_picks,
        perf=perf,
        ferment_result=ferment_result,
        psm_cross_mark=psm_cross_mark,
        psm_adjusted=psm_adjusted,
        mt_result=mt_result,
        psm_linkage_enabled=PSM_LINKAGE_ENABLED,
        system_health=system_health,
        nav_perf=nav_perf,
        ferm_hitrate=ferm_hitrate,
    )
    generator.generate(report_path)
    # 生成 HTML 版本（后台线程，不阻塞 MD 报告返回）
    def _gen_html_async(md_path):
        try:
            from report_html import md_to_html
            html_path = md_to_html(md_path)
            logger.info(f"✅ HTML 报告: {html_path}")
        except Exception as e:
            logger.warning(f"HTML 报告生成失败: {e}")
    threading.Thread(target=_gen_html_async, args=(report_path,), daemon=True).start()
    return report_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='投资建议系统')
    parser.add_argument('--date', default=None, help='目标日期 YYYY-MM-DD')
    parser.add_argument('--limit', type=int, default=300, help='股票池数量')
    parser.add_argument('--full', action='store_true', help='包含财务数据（慢）')
    args = parser.parse_args()

    target = args.date or datetime.now().strftime('%Y-%m-%d')
    path = generate_report(target, stock_limit=args.limit, skip_financials=not args.full)

    print(f"\n{'='*60}")
    with open(path, 'r') as f:
        print(f.read())
