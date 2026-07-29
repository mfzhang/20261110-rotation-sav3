"""全链路集成测试：验证 ARC+TID 双模型端到端不崩溃"""
import pytest
import pandas as pd
import numpy as np


def test_full_pipeline_smoke(synthetic_daily_df, synthetic_industry_map, synthetic_ferm_df):
    """完整管线冒烟测试：发酵检测 → 交叉验证 → 模式保存 → HTML 报告"""
    from fermentation import run_fermentation_pipeline
    from fermentation_cross import run_cross_validation
    from fermentation_patterns import save_patterns, load_patterns
    import os

    # 1. 发酵检测
    result = run_fermentation_pipeline(synthetic_daily_df, synthetic_industry_map)
    assert 'crowding' in result
    assert 'fermentation' in result
    assert 'sub_indices' in result
    assert len(result['sub_indices']) >= 3, f"子方向指数不足: {len(result['sub_indices'])}"

    ferm_df = result['fermentation']
    if ferm_df.empty:
        pytest.skip("合成数据未产生发酵候选（随机性导致），跳过后续")

    # 2. 交叉验证
    ferm_df = run_cross_validation(synthetic_daily_df, synthetic_industry_map, ferm_df)
    assert 'arc_confirm' in ferm_df.columns
    assert 'arc_position_industry' in ferm_df.columns

    # 3. 模式保存
    llm_fake = {row['sub_sector']: {
        'catalyst': 'test catalyst',
        'confirm_condition': 'test confirm',
        'fail_condition': 'test fail',
    } for _, row in ferm_df.iterrows()}
    save_patterns(ferm_df, llm_fake, '2026-06-20')
    loaded = load_patterns(days=1)
    assert '2026-06-20' in loaded

    # 4. HTML 报告
    from report_html import md_to_html
    mini_md = "# test\n## section\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    tmp_md = '/tmp/sav3/output/test_integration.md'
    tmp_html = '/tmp/sav3/output/test_integration.html'
    os.makedirs('/tmp/sav3/output', exist_ok=True)
    with open(tmp_md, 'w') as f:
        f.write(mini_md)
    html_path = md_to_html(tmp_md, tmp_html)
    assert os.path.exists(html_path)
    assert os.path.getsize(html_path) > 0
    os.remove(tmp_md)
    os.remove(tmp_html)


def test_sector_rotation_shared_data(synthetic_daily_df, synthetic_industry_map):
    """TID 板块轮动 + 共享数据管线"""
    from sector_rotation import recommend, get_stock_picks
    top3, bottom3, perf = recommend(
        daily_df=synthetic_daily_df,
        industry_map=synthetic_industry_map
    )
    assert len(top3) >= 1
    assert len(bottom3) >= 1
    picks = get_stock_picks(synthetic_daily_df, synthetic_industry_map)
    assert len(picks) >= 1


def test_no_circular_imports():
    """无循环导入"""
    import sub_sectors
    import fermentation
    import fermentation_cross
    import fermentation_patterns
    import sector_rotation
    import news_scraper
    import report_html
    import config
    import main
    assert True
