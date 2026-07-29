# SAv3 — A 股量化投资辅助系统

ARC（多因子选股）+ TID（行业轮动）+ PSM（大盘择时）三模型分立架构。

## 功能

- **ARC**：低吸 + 低波双因子选股，动态权重（崩盘加权/趋势降权），月度调仓 Top50
- **TID**：全市场 × 申万29 行业轮动，20 日夏普主信号，成交额加权
- **PSM**：六指数形态匹配 + 指标仪表盘，MA60 状态/距高点/波动率/跌幅
- **危机防御**：沪深300 20 日跌超 15% 或波动率跳升 3 倍 → 仓位降至 40%
- **日报**：每日自动生成 Markdown + HTML，含系统健康面板、净值追踪、发酵命中率

## 数据源

本系统使用 [Tushare](https://tushare.pro/) 获取 A 股行情数据。

### 配置 Tushare Token

1. 注册 [Tushare Pro](https://tushare.pro/register) 并获取 token
2. 设置环境变量或创建 `.env` 文件：

```bash
# 方式一：环境变量
export TUSHARE_TOKEN="your_token_here"

# 方式二：项目根目录 .env 文件
echo "TUSHARE_TOKEN=your_token_here" > .env
```

3. 推荐 token 档位：**2000 积分/年**（支持全市场日线 + 指数 + 股票基础信息）

## 快速开始

```bash
# 安装依赖
pip install tushare pandas numpy

# 生成日报（自动拉取数据 + 选股 + 输出）
python main.py

# 运行测试
python -m pytest tests/ -q
```

## 项目结构

```
sav3/
├── main.py              # 主编排：数据→因子→排名→日报
├── config.py            # 参数配置（因子权重/危机阈值/过滤参数）
├── data_fetcher.py      # tushare 数据管线（限速+重试+缓存）
├── data_preparation.py  # 数据清洗（涨跌停分档/流动性过滤/衍生指标）
├── factor_calculator.py # 因子计算（低吸/低波/波动率/beta）
├── score_ranker.py      # 截面 rank 归一化 + 加权排名 + IC 监控
├── sector_rotation.py   # TID 行业轮动（夏普信号/全市场成交额加权）
├── market_timing.py     # 关键点位/波浪分析 + PSM 指标仪表盘
├── scenario_generator.py# PSM 形态匹配 + 场景生成
├── scene_matcher.py     # 场景相似度匹配（特征标准化）
├── crisis_detector.py   # 危机检测器（多指数加权）
├── backtest.py          # 回测引擎（统一口径/含退市股保守口径）
├── fermentation.py      # 低位发酵选股（真实成分股广度）
├── fermentation_cross.py# ARC 发酵交叉验证
├── fermentation_patterns.py # 发酵模式数据管线
├── news_scraper.py      # 新闻抓取（6 源并发）+ 情感分析
├── portfolio.py         # 持仓分析与建议
├── report_generator.py  # Markdown 日报生成
├── report_html.py       # HTML 日报转换
├── eval_tid.py          # TID 评估脚本（生产口径 + δ 网格回测）
├── nav_tracker.py       # 纸面净值追踪
├── ferm_hitrate_analyzer.py # 发酵分档命中率分析
└── tests/               # 测试套件（64 tests）
```

## 模型说明

### ARC — 多因子选股

| 因子 | 权重 | 说明 |
|------|:----:|------|
| low_risk | 动态 | 低波动 + 低回撤 + 低 Beta 合成 |
| position | 动态 | 52 周相对低位（崩盘加权/趋势降权） |

截面 rank 归一化后加权排名，月度调仓。

### 已确认结论（2026-07 审计）

- mdd NaN 已修复（原 `cumprod` 首行 NaN 全列污染）
- 涨跌停按板块分档（主板 10%/科创创业 20%/北交 30%），tradable 标记替代删行
- TID 20 日夏普优于动量（RankIC 0.084 vs 0.074）
- PSM 指标仪表盘模式（n<20 不输出概率）
- NEWS_DELTA=0.05（δ 网格回测最优值）

## License

MIT
