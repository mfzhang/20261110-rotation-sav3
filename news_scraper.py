#!/usr/bin/env python3
# /tmp/sav3/news_scraper.py
"""财经门户爬虫：抓取头条 → 行业映射 → 情感分类 → 替代 tushare news"""

import re
import time
import json
import os
import pickle
import logging
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# ===== 行业关键词映射（29 个申万一级行业，每个 ≥6 关键词）=====
# 统一维护在 sub_sectors.py
from sub_sectors import INDUSTRY_KEYWORDS, VALID_INDUSTRIES

# ===== 情感关键词 =====
POSITIVE_WORDS = ["暴涨", "涨停", "利好", "突破", "创新高", "大增", "超预期", "政策支持",
                  "获批", "中标", "订单", "回购", "增持", "分红", "扭亏", "业绩预增",
                  "战略合作", "开放", "放宽", "取消限制", "补贴", "降息"]
NEGATIVE_WORDS = ["暴跌", "跌停", "利空", "崩盘", "亏损", "暴雷", "减持", "处罚",
                  "立案", "退市", "违约", "诉讼", "制裁", "加征关税", "贸易战",
                  "下调评级", "产能过剩", "降价", "裁员", "破产"]

# ===== 新闻源（3个可用源） =====


def fetch_sina_headlines() -> list:
    """新浪财经头条"""
    try:
        url = "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&k=&num=50&page=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read().decode())
        items = []
        for item in result.get("result", {}).get("data", [])[:50]:
            title = item.get("title", "")
            ctime = item.get("ctime", "")
            try:
                dt = datetime.fromtimestamp(int(ctime))
            except:
                dt = datetime.now()
            items.append({"title": title, "time": dt, "source": "新浪财经"})
        return items
    except Exception as e:
        print(f"  [WARN] 新浪财经拉取失败: {e}")
        return []


def fetch_10jqka_headlines() -> list:
    """同花顺快讯（个股公告）"""
    try:
        url = "https://news.10jqka.com.cn/tapp/news/push/stock/?page=1&tag="
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read().decode())
        items = []
        for item in result.get("data", {}).get("list", [])[:20]:
            title = item.get("title", "")
            ctime = item.get("ctime", "")
            try:
                dt = datetime.fromtimestamp(int(ctime))
            except:
                dt = datetime.now()
            items.append({"title": title, "time": dt, "source": "同花顺"})
        return items
    except Exception as e:
        print(f"  [WARN] 同花顺拉取失败: {e}")
        return []


def fetch_wallstcn_headlines() -> list:
    """华尔街见闻（全球宏观）"""
    try:
        url = "https://api-one.wallstcn.com/apiv1/content/lives?channel=global-channel&client=pc&limit=50&first_page=true"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read().decode())
        items = []
        for item in result.get("data", {}).get("items", [])[:50]:
            title = item.get("content_text", "")
            ctime = item.get("display_time", 0)
            try:
                dt = datetime.fromtimestamp(int(ctime))
            except:
                dt = datetime.now()
            items.append({"title": title, "time": dt, "source": "华尔街见闻"})
        return items
    except Exception as e:
        print(f"  [WARN] 华尔街见闻拉取失败: {e}")
        return []


def fetch_jin10_headlines() -> list:
    """金十数据快讯"""
    try:
        url = "https://flash-api.jin10.com/get_flash_list?channel=-8200&vip=1"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.jin10.com/",
            "x-app-id": "bVBF4FyRTn5NJF5n",
            "x-version": "1.0.0"
        })
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read().decode())
        items = []
        for item in result.get("data", [])[:40]:
            # type=0 普通快讯有 content；type=1/2 头条有 title
            inner = item.get("data", {})
            title = inner.get("title", "") or inner.get("content", "")
            if not title:
                continue
            time_str = item.get("time", "")
            try:
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except:
                dt = datetime.now()
            items.append({"title": title, "time": dt, "source": "金十数据"})
        return items
    except Exception as e:
        print(f"  [WARN] 金十数据拉取失败: {e}")
        return []


def fetch_yicai_headlines() -> list:
    """第一财经新闻"""
    try:
        url = "https://www.yicai.com/api/ajax/getlistbycid?cid=49&page=1&pagesize=30"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read().decode())
        items = []
        for item in result[:30]:
            title = item.get("NewsTitle", "")
            if not title:
                continue
            time_str = item.get("CreateDate", "")
            try:
                dt = datetime.strptime(time_str[:19], "%Y-%m-%dT%H:%M:%S")
            except:
                dt = datetime.now()
            items.append({"title": title, "time": dt, "source": "第一财经"})
        return items
    except Exception as e:
        print(f"  [WARN] 第一财经拉取失败: {e}")
        return []


def fetch_163_headlines() -> list:
    """网易财经头条"""
    try:
        url = "https://money.163.com/special/00251G8F/news_json.js"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode("utf-8", errors="replace")
        # 解析 JavaScript 对象格式: {c:0,t:"标题",l:"链接",p:"时间"}
        pattern = r'\{c:\d+,t:"([^"]+)",l:"([^"]+)",p:"([^"]+)"\}'
        items = []
        today = datetime.now().strftime("%Y-%m-%d")
        for m in re.finditer(pattern, raw):
            title = m.group(1)
            time_str = m.group(3)
            # 只取今日新闻
            if not time_str.startswith(today):
                continue
            try:
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except:
                dt = datetime.now()
            items.append({"title": title, "time": dt, "source": "网易财经"})
        return items
    except Exception as e:
        print(f"  [WARN] 网易财经拉取失败: {e}")
        return []


# ===== 6 源并发抓取 + 当日缓存 =====
NEWS_SOURCES = [
    ("新浪财经", fetch_sina_headlines),
    ("同花顺", fetch_10jqka_headlines),
    ("华尔街见闻", fetch_wallstcn_headlines),
    ("金十数据", fetch_jin10_headlines),
    ("第一财经", fetch_yicai_headlines),
    ("网易财经", fetch_163_headlines),
]

_NEWS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(_NEWS_CACHE_DIR, exist_ok=True)


def _fetch_all_headlines(target_date: str) -> list:
    """6 个新闻源并发抓取头条；当日结果写磁盘缓存，同日重复调用直接命中。

    缓存文件名含日期（news_headlines_YYYY-MM-DD.pkl），天然按日隔离；
    命中条件为目标日期 == 今天，历史日期（回测）不读缓存。
    """
    today = datetime.now().strftime('%Y-%m-%d')
    cache_name = f'news_headlines_{target_date}.pkl'
    cache_path = os.path.join(_NEWS_CACHE_DIR, cache_name)

    if target_date == today and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            print(f"  命中当日新闻缓存 ({len(cached)} 条)")
            return cached
        except Exception:
            pass  # 缓存损坏则重新抓取

    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=len(NEWS_SOURCES)) as ex:
        future_to_name = {ex.submit(func): name for name, func in NEWS_SOURCES}
        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            try:
                results[name] = fut.result()
            except Exception as e:
                print(f"  [WARN] {name}拉取异常: {e}")
                results[name] = []

    # 按 NEWS_SOURCES 固定顺序汇总，保证输出可复现
    all_items = []
    parts = []
    for name, _ in NEWS_SOURCES:
        items = results.get(name) or []
        all_items.extend(items)
        parts.append(f"{name}={len(items)}")
    print(f"  各源条数: {' '.join(parts)} → 合计={len(all_items)} "
          f"(并发耗时 {time.time() - t0:.1f}s)")

    # 仅缓存当日抓取结果，并清理非当日缓存文件
    if target_date == today and all_items:
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(all_items, f)
        except Exception:
            pass
        for f_ in os.listdir(_NEWS_CACHE_DIR):
            if f_.startswith('news_headlines_') and f_ != cache_name:
                try:
                    os.remove(os.path.join(_NEWS_CACHE_DIR, f_))
                except OSError:
                    pass
    return all_items


def map_to_industry(title: str) -> list:
    """标题 → 行业列表（降级路径）"""
    industries = set()
    for ind, keywords in INDUSTRY_KEYWORDS.items():
        for kw in keywords:
            if kw in title:
                industries.add(ind)
                break
    return list(industries) if industries else []


def classify_sentiment(title: str) -> int:
    """标题情感分类 +1/-1/0（降级路径）"""
    pos = sum(1 for w in POSITIVE_WORDS if w in title)
    neg = sum(1 for w in NEGATIVE_WORDS if w in title)
    if pos > neg:
        return 1
    elif neg > pos:
        return -1
    return 0


def _classify_batch_keywords(headlines: list) -> list:
    """降级路径：用关键词法批量分类（情感 + 行业映射）。

    参数:
        headlines: list of dict，每个元素含有 title, source, time
    返回:
        list of dict [{title, time, sentiment, industries}, ...]
        跳过 sentiment==0 或 industries 为空的条目。
    """
    results = []
    for item in headlines:
        title = item.get("title", "")
        sentiment = classify_sentiment(title)
        if sentiment == 0:
            continue
        industries = map_to_industry(title)
        if not industries:
            continue
        results.append({
            "title": title,
            "time": item.get("time"),
            "sentiment": sentiment,
            "industries": industries,
        })
    return results


# ===================== 新增：DeepSeek API 情感分类 =====================
def _call_deepseek_sentiment_batched(headlines: list, batch_size: int = 60) -> list | None:
    """
    分批调用 DeepSeek 对头条列表进行情感/行业分类。

    参数:
        headlines: list of dict，每个元素含有 title, source, time
        batch_size: 每批最多条数，默认 60
    返回:
        成功 -> 返回 list of dict [{title, time, sentiment, confidence, industries}, ...]
        所有批次均失败 -> None（触发降级路径）
    """
    total = len(headlines)
    batches = [headlines[i:i + batch_size] for i in range(0, total, batch_size)]
    logging.info(f"DeepSeek 分批分类: {total}条 → {len(batches)}批 (batch_size={batch_size})")

    all_results = []
    for batch_idx, batch in enumerate(batches):
        logging.info(f"  批次 {batch_idx + 1}/{len(batches)}: {len(batch)}条")
        batch_result = _call_deepseek_sentiment(batch)
        if batch_result is None:
            logging.warning(f"  批次 {batch_idx + 1}/{len(batches)} 失败，跳过")
            continue
        all_results.extend(batch_result)
        logging.info(f"  批次 {batch_idx + 1}/{len(batches)} 成功: {len(batch_result)}条")

    if not all_results:
        logging.warning("所有批次均失败，触发降级路径")
        return None

    logging.info(f"分批分类完成: 成功 {len(all_results)}/{total} 条")
    return all_results


def _call_deepseek_sentiment(headlines: list) -> list | None:
    """
    调用 DeepSeek 对头条列表进行情感/行业分类。

    参数:
        headlines: list of dict，每个元素含有 title, source, time
    返回:
        成功 -> 返回 list of dict [{title, time, sentiment, confidence, industries}, ...]
        失败 -> None
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        # 从 .env 文件兜底读取（cron 环境可能没加载）
        env_file = os.path.join(os.path.dirname(__file__), '..', '.env')
        for candidate in [
            os.path.expanduser('~/.hermes/profiles/sa-system/.env'),
            '/root/.hermes/profiles/sa-system/.env',
        ]:
            if os.path.exists(candidate):
                env_file = candidate
                break
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('DEEPSEEK_API_KEY='):
                        api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass
    if not api_key:
        logging.warning("API分类失败(原因:未设置环境变量DEEPSEEK_API_KEY)，降级关键词法")
        return None

    # 构造 user message
    user_lines = []
    for i, item in enumerate(headlines):
        src = item.get("source", "未知")
        title = item.get("title", "").replace("\n", "").replace("\r", "")
        user_lines.append(f"{i}|{src}|{title}")
    user_message = "\n".join(user_lines)

    system_prompt = (
        "你是A股财经新闻情感分析器。对每条新闻判断其对股价的影响方向。\n\n"
        "输出纯JSON数组，无解释文字、无代码块包裹：\n"
        '[{"idx":0,"sentiment":-1,"confidence":0.9,"industries":["银行","房地产"]},'
        '{"idx":1,"sentiment":1,"confidence":0.7,"industries":["电子"]}]\n\n'
        "字段规则：\n"
        "- idx: 输入序号\n"
        "- sentiment: -1=利空, 0=中性, 1=利好\n"
        "- confidence: 0.0-1.0，你的确信程度\n"
        "- industries: 从以下选：["
        "银行,非银金融,房地产,电子,计算机,电力设备,汽车,医药生物,食品饮料,国防军工,通信,有色金属,煤炭,石油石化,"
        "建筑装饰,传媒,社会服务,商贸零售,交通运输,公用事业,环保,钢铁,基础化工,机械设备,家用电器,纺织服饰,农林牧渔,"
        "轻工制造,综合"
        "]\n\n"
        "判断参考（非穷举）：\n"
        "- 降准/降息/减税/补贴/放开/中标/回购/增持/分红 → 利好\n"
        "- 加息/处罚/立案/减持/违约/制裁/加税/退市/裁员 → 利空\n"
        "- 定期报告/人事变动/例行会议 → 中性"
    )

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0,
        "max_tokens": 4096
    }

    try:
        req = urllib.request.Request(
            "https://api.deepseek.com/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            if resp.status != 200:
                logging.warning(f"API分类失败(原因:HTTP状态码{resp.status})，降级关键词法")
                return None
            raw = resp.read().decode()
            if not raw:
                logging.warning("API分类失败(原因:响应体为空)，降级关键词法")
                return None
            result = json.loads(raw)
    except urllib.error.HTTPError as e:
        logging.warning(f"API分类失败(原因:HTTP状态码{e.code})，降级关键词法")
        return None
    except urllib.error.URLError as e:
        logging.warning(f"API分类失败(原因:网络错误/{e.reason})，降级关键词法")
        return None
    except json.JSONDecodeError:
        logging.warning("API分类失败(原因:响应JSON解析失败)，降级关键词法")
        return None
    except Exception as e:
        logging.warning(f"API分类失败(原因:{e})，降级关键词法")
        return None

    # 提取 assistant 回复
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logging.warning("API分类失败(原因:响应结构异常，无choices[0].message.content)，降级关键词法")
        return None

    # 解析 JSON 数组（允许最外层有 markdown 代码块包裹）
    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # 尝试去掉可能的 markdown 代码块标记
        cleaned = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.IGNORECASE)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logging.warning("API分类失败(原因:JSON解析失败，且strip markdown后仍失败)，降级关键词法")
            return None

    if not isinstance(parsed, list):
        logging.warning("API分类失败(原因:返回非数组)，降级关键词法")
        return None
    if len(parsed) != len(headlines):
        logging.warning(f"API分类失败(原因:返回数量{len(parsed)} ≠ 输入{len(headlines)})，降级关键词法")
        return None

    # 逐条校验
    for item in parsed:
        if not isinstance(item, dict):
            logging.warning("API分类失败(原因:数组元素非对象)，降级关键词法")
            return None
        idx = item.get("idx")
        sentiment = item.get("sentiment")
        confidence = item.get("confidence")
        industries = item.get("industries")
        if idx is None or not isinstance(idx, int) or idx < 0 or idx >= len(headlines):
            logging.warning("API分类失败(原因:非法idx)，降级关键词法")
            return None
        if sentiment not in (-1, 0, 1):
            logging.warning("API分类失败(原因:sentiment值非法)，降级关键词法")
            return None
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            logging.warning("API分类失败(原因:confidence值非法)，降级关键词法")
            return None
        if not isinstance(industries, list):
            logging.warning("API分类失败(原因:industries非数组)，降级关键词法")
            return None
        # 过滤非法行业名，不导致整批失败；单个行业非法时跳过，不加入该条目
        valid_inds = [ind for ind in industries if ind in VALID_INDUSTRIES]
        if len(valid_inds) < len(industries):
            skipped = [ind for ind in industries if ind not in VALID_INDUSTRIES]
            logging.info(f"跳过非法行业名称: {skipped}")
        # 将过滤后的行业列表写回，后续组装时直接使用
        item['industries'] = valid_inds

    # 组装结果，保留原有 title 等信息方便后续聚合
    result_list = []
    for i, item in enumerate(parsed):
        src_head = headlines[i]
        result_list.append({
            "title": src_head["title"],
            "time": src_head["time"],
            "source": src_head.get("source", "未知"),
            "sentiment": item["sentiment"],
            "confidence": item["confidence"],
            "industries": item["industries"]
        })
    return result_list


# ===================== 主入口：scrape_daily_news =====================
def scrape_daily_news(target_date: str = None) -> pd.DataFrame:
    """
    爬取当日财经头条，返回行业级情绪。
    DataFrame[date, industry, sentiment_raw, confidence]
    """
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')

    print(f"爬取 {target_date} 财经头条...")

    all_items = _fetch_all_headlines(target_date)
    if not all_items:
        print("  ⚠️ 无新闻数据")
        return pd.DataFrame(columns=['date', 'industry', 'sentiment_raw', 'confidence'])

    # 尝试 API 情感分类
    t0 = time.time()
    api_results = _call_deepseek_sentiment_batched(all_items, batch_size=60)
    if api_results is not None:
        elapsed = time.time() - t0
        logging.info(f"DeepSeek 情感分类: {len(all_items)}条头条, 耗时{elapsed:.1f}s")
        # 用 API 结果构建 DataFrame
        records = []
        for item in api_results:
            news_date = item['time'].date()  # 用每条新闻自身的发布时间
            for ind in item['industries']:
                records.append({
                    'date': news_date,
                    'industry': ind,
                    'sentiment': item['sentiment'],
                    'confidence': item['confidence']
                })
        if not records:
            return pd.DataFrame(columns=['date', 'industry', 'sentiment_raw', 'confidence'])
        df = pd.DataFrame(records)
        agg = df.groupby(['date', 'industry'], as_index=False).agg(
            sentiment_raw=('sentiment', 'mean'),
            confidence=('confidence', 'mean')
        )
        print(f"  行业信号(API): {len(agg)} 条, {agg['industry'].nunique()} 个行业")
        return agg[['date', 'industry', 'sentiment_raw', 'confidence']]

    # ---- 降级路径：关键词法批量分类 ----
    kw_results = _classify_batch_keywords(all_items)
    records = []
    target_date_dt = pd.to_datetime(target_date)
    for item in kw_results:
        t = item.get('time')
        news_date = t.date() if t else target_date_dt  # 时间缺失时 fallback
        for ind in item['industries']:
            records.append({
                'date': news_date,
                'industry': ind,
                'sentiment': item['sentiment'],
            })

    if not records:
        return pd.DataFrame(columns=['date', 'industry', 'sentiment_raw', 'confidence'])

    df = pd.DataFrame(records)

    # 聚合到行业级
    agg = df.groupby(['date', 'industry'], as_index=False).agg(
        sentiment_raw=('sentiment', 'mean'),
        event_count=('sentiment', 'count')
    )

    threshold = 3
    agg['confidence'] = np.minimum(agg['event_count'] / threshold, 1.0)

    print(f"  行业信号(降级): {len(agg)} 条, {agg['industry'].nunique()} 个行业")
    return agg[['date', 'industry', 'sentiment_raw', 'confidence']]


# ===================== 新增：详细新闻（保留标题 + 时效加权 + 子方向分类） =====================
def _compute_time_decay(news_time: datetime, reference_time: datetime = None) -> float:
    """时效性加权：越新权重越高"""
    if reference_time is None:
        reference_time = datetime.now()
    hours = (reference_time - news_time).total_seconds() / 3600
    if hours <= 2:
        return 1.0
    elif hours <= 6:
        return 0.8
    elif hours <= 12:
        return 0.6
    elif hours <= 24:
        return 0.4
    elif hours <= 48:
        return 0.2
    else:
        return 0.1


def scrape_daily_news_detailed(target_date: str = None) -> dict:
    """
    爬取财经头条，返回详细结果（保留标题、时效加权、子方向分类）。

    返回: {
        'aggregated': DataFrame (同 scrape_daily_news 格式, 行业级聚合),
        'headlines': list of dict [
            {title, time, source, sentiment, confidence, industries, sub_sectors, decay_weight}
        ],
        'total_count': int,
        'api_success': bool,
    }
    """
    from sub_sectors import classify_news_to_sub

    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')

    print(f"爬取 {target_date} 财经头条（详细模式）...")

    all_items = _fetch_all_headlines(target_date)

    if not all_items:
        return {
            'aggregated': pd.DataFrame(columns=['date', 'industry', 'sentiment_raw', 'confidence']),
            'headlines': [],
            'total_count': 0,
            'api_success': False,
        }

    # 保存 source 映射（API 路径回填用；优先使用 API 返回的 source，避免分批索引错位）
    source_map = {}
    for i, item in enumerate(all_items):
        source_map[i] = item.get('source', '未知')

    # API 情感分类
    t0 = time.time()
    api_results = _call_deepseek_sentiment_batched(all_items, batch_size=60)
    api_success = api_results is not None

    if api_results is not None:
        elapsed = time.time() - t0
        logging.info(f"DeepSeek 情感分类: {len(all_items)}条, 耗时{elapsed:.1f}s")

        # 构建详细头条列表
        headlines = []
        ref_time = datetime.now()
        for idx, item in enumerate(api_results):
            news_time = item['time']
            if isinstance(news_time, datetime):
                decay = _compute_time_decay(news_time, ref_time)
            else:
                decay = 0.5
            sub_sectors = classify_news_to_sub(item['title'])
            headlines.append({
                'title': item['title'],
                'time': news_time,
                'source': item.get('source', source_map.get(idx, '未知')),
                'sentiment': item['sentiment'],
                'confidence': item['confidence'],
                'industries': item['industries'],
                'sub_sectors': sub_sectors,
                'decay_weight': decay,
            })

        # 行业级聚合（同理现有逻辑）
        records = []
        for item in api_results:
            news_date = item['time'].date() if isinstance(item['time'], datetime) else pd.to_datetime(target_date).date()
            for ind in item['industries']:
                records.append({
                    'date': news_date,
                    'industry': ind,
                    'sentiment': item['sentiment'],
                    'confidence': item['confidence'],
                })
        if not records:
            agg = pd.DataFrame(columns=['date', 'industry', 'sentiment_raw', 'confidence'])
        else:
            agg = pd.DataFrame(records).groupby(['date', 'industry'], as_index=False).agg(
                sentiment_raw=('sentiment', 'mean'),
                confidence=('confidence', 'mean')
            )

        # 子方向级聚合
        sub_records = []
        for h in headlines:
            news_date = h['time'].date() if isinstance(h['time'], datetime) else pd.to_datetime(target_date).date()
            for sub in h['sub_sectors']:
                sub_records.append({
                    'date': news_date,
                    'sub_sector': sub,
                    'sentiment': h['sentiment'],
                    'confidence': h['confidence'],
                    'decay_weight': h['decay_weight'],
                })
        sub_agg = pd.DataFrame(sub_records) if sub_records else pd.DataFrame(
            columns=['date', 'sub_sector', 'sentiment', 'confidence', 'decay_weight'])

        return {
            'aggregated': agg[['date', 'industry', 'sentiment_raw', 'confidence']],
            'headlines': headlines,
            'sub_aggregated': sub_agg,
            'total_count': len(all_items),
            'api_success': True,
        }

    # ---- 降级路径 ----
    kw_results = _classify_batch_keywords(all_items)
    headlines = []
    ref_time = datetime.now()
    for idx, item in enumerate(kw_results):
        t = item.get('time')
        decay = _compute_time_decay(t, ref_time) if isinstance(t, datetime) else 0.5
        sub_sectors = classify_news_to_sub(item['title'])
        headlines.append({
            'title': item['title'],
            'time': t,
            'source': source_map.get(idx, '未知'),
            'sentiment': item['sentiment'],
            'confidence': 0.5,  # keyword-based: lower confidence
            'industries': item['industries'],
            'sub_sectors': sub_sectors,
            'decay_weight': decay,
        })

    records = []
    target_date_dt = pd.to_datetime(target_date)
    for item in kw_results:
        t = item.get('time')
        news_date = t.date() if t else target_date_dt
        for ind in item['industries']:
            records.append({
                'date': news_date,
                'industry': ind,
                'sentiment': item['sentiment'],
            })

    if not records:
        agg = pd.DataFrame(columns=['date', 'industry', 'sentiment_raw', 'confidence', 'event_count'])
    else:
        agg = pd.DataFrame(records).groupby(['date', 'industry'], as_index=False).agg(
            sentiment_raw=('sentiment', 'mean'),
            event_count=('sentiment', 'count')
        )
        agg['confidence'] = np.minimum(agg['event_count'] / 3, 1.0)

    # 子方向聚合
    sub_records = []
    for h in headlines:
        news_date = h['time'].date() if isinstance(h['time'], datetime) else target_date_dt
        for sub in h['sub_sectors']:
            sub_records.append({
                'date': news_date,
                'sub_sector': sub,
                'sentiment': h['sentiment'],
                'confidence': h['confidence'],
                'decay_weight': h['decay_weight'],
            })
    sub_agg = pd.DataFrame(sub_records) if sub_records else pd.DataFrame(
        columns=['date', 'sub_sector', 'sentiment', 'confidence', 'decay_weight'])

    print(f"  行业信号(降级): {len(agg)} 条, {agg['industry'].nunique()} 个行业")
    return {
        'aggregated': agg[['date', 'industry', 'sentiment_raw', 'confidence']],
        'headlines': headlines,
        'sub_aggregated': sub_agg,
        'total_count': len(all_items),
        'api_success': False,
    }


if __name__ == '__main__':
    result = scrape_daily_news()
    if not result.empty:
        print("\n=== 今日行业情绪 ===")
        print(result.sort_values('sentiment_raw', ascending=False).head(10).to_string(index=False))
