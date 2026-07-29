import os
import warnings

# --- 通用 API Key 读取：环境变量 > .env 文件 > default ---
def load_api_key(key_name: str, default: str = '') -> str:
    """
    读取 API Key：环境变量 > 项目根目录/用户目录 .env 文件 > default。
    支持从环境变量、.env 文件中按 key_name=VALUE 格式读取。
    """
    # 1. 环境变量
    value = os.environ.get(key_name, '')
    if value:
        return value

    # 2. .env 文件候选路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, '.env'),
        os.path.join(base_dir, '..', '.env'),
        os.path.expanduser('~/.hermes/profiles/sa-system/.env'),
        '/root/.hermes/profiles/sa-system/.env',
    ]
    for env_file in candidates:
        if os.path.exists(env_file):
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        k, v = line.split('=', 1)
                        if k.strip() == key_name:
                            return v.strip().strip('"').strip("'")
            except Exception:
                pass

    return default


# --- Tushare Token: 优先从环境变量/ .env 读取，fallback 到默认值 ---
def _load_token():
    """读取 Tushare Token：环境变量 > .env 文件 > 默认值"""
    token = load_api_key('TUSHARE_TOKEN')
    if token:
        return token
    # 降级：返回空 token，后续 tushare 调用自行报错但不崩溃整个导入链
    warnings.warn("TUSHARE_TOKEN 未设置。请设置环境变量 TUSHARE_TOKEN 或在项目目录创建 .env 文件。tushare 接口将不可用。")
    return ''

TUSHARE_TOKEN = _load_token()

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# 精简后因子权重（审计后移除 momentum/rel_strength）
# 注意：正常模式下实际权重由 compute_regime_weights 按市场状态动态调整（P1-6），
# 此处为中性基准值；崩盘/趋势市的偏移见下方 REGIME_* 参数
FACTOR_WEIGHTS = {
    'low_risk': 0.50,
    'position': 0.50,
}

# position 因子动态配权（P1-6，review §3.1 实证：position 是"崩盘/反转期权"——
# 崩盘月 RankIC +0.54，趋势市连续 -0.14~-0.18；固定 50% 权重在趋势市受损）
DYNAMIC_FACTOR_WEIGHTS = True
REGIME_CRASH_RET20 = -0.08    # 指数20日跌幅 ≤ -8% → 崩盘/超跌状态
REGIME_CRASH_DD60 = -0.12     # 或距60日高点回撤 ≤ -12%
REGIME_TREND_RET20 = 0.05     # 指数20日涨幅 ≥ +5% 且站上MA60 → 趋势市
REGIME_CRASH_WEIGHTS = {'low_risk': 0.35, 'position': 0.65}   # 崩盘：反转因子加权
REGIME_TREND_WEIGHTS = {'low_risk': 0.65, 'position': 0.35}   # 趋势：position 降权

# 因子列顺序（用于输出表格，保留扩展性——未来加新因子时追加到这里）
FACTOR_ORDER = ['low_risk', 'position']

# 危机防御权重
CRISIS_WEIGHTS = {
    'low_risk': 1.0,
    'position': 0.0,
}

# 危机参数
CRISIS_IDX_DRAWDOWN = -0.15    # 指数20日跌超15%
CRISIS_VOL_RATIO = 3.0          # 波动率跳升3倍
CRISIS_EXIT_DRAWDOWN = -0.10    # 修复到-10%以内
CRISIS_EXIT_DAYS = 5            # 连续5天无触发才退出
CRISIS_EXPOSURE = 0.4         # 危机仓位降至40%
CRISIS_EXIT_VOL_RATIO = 2.0    # 退出危机需波动率回落到2.0x以内（1.5 在波动中枢上移的市场会卡死退出，实证见 review_2026-07-18 §2.2）
CRISIS_COOLING_DAYS = 10        # 危机冷却天数
# 新闻调节系数（控制新闻情感对行业动量的影响力）
NEWS_DELTA = 0.05  # δ网格回测 2026-07-19 最优（eval_tid.py --grid-delta，代理情绪口径）
CRISIS_STOCK_STOPLOSS = -0.15   # 个股止损线

# 财务数据总开关：tushare income/balancesheet/cashflow 尚未接入（见 data_preparation TODO）。
# 显式关闭；为 False 时 prepare_data 忽略 skip_financials=False（--full），一律走 quick 模式，
# 避免调用方误以为传 False 就能拿到 np_margin/cash_to_np/leverage 财务因子。
FINANCIAL_DATA_ENABLED = False

# 过滤参数
MIN_LIST_DAYS = 60
LIQUIDITY_WINDOW = 20
LIQUIDITY_PERCENTILE = 0.1
LIQUIDITY_MIN_AMOUNT = 5_000          # 日均成交额门槛（千元，500万元=5000千元）
STOP_FILTER = True

# 因子计算窗口
REL_STRENGTH_DAYS = 20
MOMENTUM_WINDOW = 252
MOMENTUM_SKIP = 20
LOW_RISK_WINDOW = 252
HIGH_LOW_WINDOW = 252

BACKTEST_TRADING_COST = 0.00025   # 单边佣金 万2.5 (0.025%)
BACKTEST_STAMP_DUTY = 0.0005   # 印花税 0.05%（卖出收取）
BACKTEST_SLIPPAGE = 0.001      # 滑点+冲击成本 0.1%（单边，含小盘股冲击）
TOP_N = 50
INDEX_CODE = 'sh.000300'

# P1修复：多指数加权危机检测（解决成长策略用价值指数做对冲的错配）
CRISIS_INDEX_CODES = ['000300.SH', '399006.SZ', '000688.SH']  # 沪深300, 创业板指, 科创50
CRISIS_INDEX_WEIGHTS = [0.4, 0.3, 0.3]  # 加权权重

# 回测成本模型假设
TURNOVER_RATE = 1.0            # 换手率假设=100%（简化假设，未纳入成本计算）
