#!/usr/bin/env python3
# /tmp/sav3/portfolio.py
# 持仓分析 — 读 holdings.json，取当前价，给建议
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("portfolio")

HOLDINGS_FILE = os.path.join(os.path.dirname(__file__), 'holdings.json')


def load_holdings() -> list:
    """加载持仓 [{code, name, shares, cost, date}]"""
    if not os.path.exists(HOLDINGS_FILE):
        raise FileNotFoundError(f"持仓文件 {HOLDINGS_FILE} 不存在，请先创建")
    with open(HOLDINGS_FILE, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'holdings' in data:
        return data['holdings']
    raise ValueError("持仓格式错误")


def analyze_holdings(prices_override: dict = None) -> list:
    """
    返回分析结果 [{code, name, cost, current, shares, pct_change, advice}]

    prices_override: {code: float} — tushare 收盘价（日报流程由 main.py 经
        daily_df + fetch_holdings_closes 小缓存全覆盖后传入）。
        未覆盖的 code 不再降级新浪实时接口（sina_api 路径已移除），
        直接标记「无法获取价格」。
    """
    holdings = load_holdings()
    if not holdings:
        logger.warning("持仓为空")
        return []

    codes = [h['code'] for h in holdings]

    # 价格唯一来源：prices_override（tushare 收盘价）
    prices = {}
    if prices_override:
        for code in codes:
            if code in prices_override:
                prices[code] = {"price": prices_override[code], "volume": 0}
    missing = [c for c in codes if c not in prices]
    if missing:
        logger.warning(f"持仓价格未覆盖（无 tushare 收盘价）: {missing}")

    results = []
    for h in holdings:
        code = h['code']
        cost = h.get('cost', 0)
        shares = h.get('shares', 0)
        name = h.get('name', code)
        quote = prices.get(code)
        
        if quote is None:
            results.append({
                'code': code, 'name': name, 'cost': cost,
                'current': 'N/A', 'shares': shares,
                'pct_change': None, 'advice': '无法获取价格'
            })
            continue
        
        current = quote.get('price')
        volume = quote.get('volume', 0)
        
        if current is None or cost == 0:
            results.append({
                'code': code, 'name': name, 'cost': cost,
                'current': current or 'N/A', 'shares': shares,
                'pct_change': None, 'advice': '无法获取价格'
            })
            continue
        
        # 停牌检测：价格≈0且成交量为0
        if abs(current) < 0.01 and volume == 0:
            results.append({
                'code': code, 'name': name, 'cost': cost,
                'current': current, 'shares': shares,
                'pct_change': None, 'advice': '🛑 停牌中（无法交易）'
            })
            continue
        
        pct = round((current - cost) / cost * 100, 2)
        
        if pct <= -20:
            advice = '⚠️ 建议清仓（亏损超20%）'
        elif pct <= -10:
            advice = '🔻 建议减仓（亏损超10%）'
        elif pct >= 30:
            advice = '💰 建议止盈（盈利超30%）'
        else:
            advice = '✅ 继续持有'
        
        results.append({
            'code': code, 'name': name, 'cost': cost,
            'current': current, 'shares': shares,
            'pct_change': pct, 'advice': advice
        })
    
    return results


if __name__ == '__main__':
    try:
        results = analyze_holdings()
        for r in results:
            print(f"{r['name']}({r['code']}): "
                  f"成本{r['cost']} 现价{r['current']} "
                  f"盈亏{r['pct_change']}% → {r['advice']}")
    except Exception as e:
        logger.error(f"执行失败: {e}")
