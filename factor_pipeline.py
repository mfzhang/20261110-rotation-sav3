import pandas as pd
import numpy as np

def winsorize(series, limits=(0.01,0.99)):
    s = series.dropna()
    if s.empty: return series
    return series.clip(s.quantile(limits[0]), s.quantile(limits[1]))

def neutralize(series, industry_series):
    df = pd.DataFrame({'f':series,'ind':industry_series}).dropna()
    if df.empty or df['ind'].nunique()<2: return series
    dummies = pd.get_dummies(df['ind'], drop_first=False).astype(float)
    try:
        beta = np.linalg.lstsq(dummies.values, df['f'].values, rcond=None)[0]
        res = pd.Series(df['f'].values - dummies.values@beta, index=df.index, dtype=float)
        result = pd.Series(index=series.index, dtype=float)
        result.loc[res.index] = res.values
        return result
    except: return series

# gram_schmidt() 已移除 — 审计确认仅2因子时正交化扭曲经济含义，不再使用
# 未来若因子数≥5，可从 git history 恢复此函数

def pipeline_process(factors_df, industry_series):
    processed = []
    for dt in factors_df.index.get_level_values('date').unique():
        sub = factors_df.xs(dt, level='date').copy()
        if sub.empty: continue
        for c in sub.columns:
            sub[c] = winsorize(sub[c])
        for c in sub.columns:
            sub[c] = neutralize(sub[c], industry_series.reindex(sub.index))
        for c in sub.columns:
            s = sub[c].dropna()
            sub[c] = (sub[c]-s.mean())/(s.std()+1e-10) if len(s)>1 and s.std()>0 else 0
        # 跳过 Gram-Schmidt 正交化（仅2因子时正交化扭曲因子经济含义）
        sub['date'] = dt
        sub['code'] = sub.index
        processed.append(sub)
    return pd.concat(processed, ignore_index=True).set_index(['date','code']).sort_index()
