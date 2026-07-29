"""单元测试：sub_sectors.py 数据完整性"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sub_sectors import (
    SUB_TO_SHENWAN,
    SHENWAN_TO_SUB,
    SUB_KEYWORDS,
    STOCK_TO_SUB_KEYWORDS,
    TUSHARE_TO_SHENWAN,
    INDUSTRY_KEYWORDS,
    VALID_INDUSTRIES,
    get_sub_sectors,
    classify_news_to_sub,
    get_all_sub_sectors,
)


class TestDataIntegrity:
    """数据完整性测试"""

    def test_total_industries(self):
        """SHENWAN_TO_SUB 有 29 个 key（申万一级行业全覆盖）"""
        assert len(SHENWAN_TO_SUB) == 29, (
            f"Expected 29 industries, got {len(SHENWAN_TO_SUB)}: "
            f"{sorted(SHENWAN_TO_SUB.keys())}"
        )

    def test_total_sub_sectors(self):
        """get_all_sub_sectors() 返回 ≥95 个子方向"""
        all_subs = get_all_sub_sectors()
        assert len(all_subs) >= 95, (
            f"Expected ≥95 sub-sectors, got {len(all_subs)}"
        )

    def test_no_duplicates(self):
        """所有子方向名唯一"""
        all_subs = get_all_sub_sectors()
        duplicates = [s for s in all_subs if all_subs.count(s) > 1]
        assert len(set(all_subs)) == len(all_subs), (
            f"Duplicate sub-sector names found: {list(set(duplicates))}"
        )

    def test_bijection(self):
        """SUB_TO_SHENWAN 和 SHENWAN_TO_SUB 互为逆映射"""
        # 每个 sub → shenwan 映射的 shenwan 应包含该 sub
        for sub_name, shenwan in SUB_TO_SHENWAN.items():
            assert shenwan in SHENWAN_TO_SUB, (
                f"SUB_TO_SHENWAN['{sub_name}'] = '{shenwan}' "
                f"but '{shenwan}' not in SHENWAN_TO_SUB"
            )
            assert sub_name in SHENWAN_TO_SUB[shenwan], (
                f"'{sub_name}' not found in SHENWAN_TO_SUB['{shenwan}']"
            )

        # 每个 shenwan → [subs] 的每个 sub 应反查回该 shenwan
        for shenwan, subs in SHENWAN_TO_SUB.items():
            for sub_name in subs:
                assert sub_name in SUB_TO_SHENWAN, (
                    f"'{sub_name}' from SHENWAN_TO_SUB['{shenwan}'] "
                    f"not in SUB_TO_SHENWAN"
                )
                assert SUB_TO_SHENWAN[sub_name] == shenwan, (
                    f"SUB_TO_SHENWAN['{sub_name}'] = "
                    f"'{SUB_TO_SHENWAN[sub_name]}', expected '{shenwan}'"
                )

    def test_news_classify_empty(self):
        """classify_news_to_sub('') 返回空列表"""
        result = classify_news_to_sub("")
        assert result == [], f"Expected [], got {result}"

    def test_news_classify_known(self):
        """classify_news_to_sub('储能项目投运') 返回 ['储能']"""
        result = classify_news_to_sub("储能项目投运")
        assert result == ["储能"], (
            f"Expected ['储能'], got {result}"
        )

    def test_news_classify_multiple(self):
        """一条新闻可以匹配多个子方向"""
        # "储能" 和 "光伏" 都是电力设备下的子方向关键词
        result = classify_news_to_sub("储能和光伏项目同步推进")
        assert "储能" in result, f"'储能' not in result: {result}"
        assert "光伏" in result, f"'光伏' not in result: {result}"
        assert len(result) >= 2, (
            f"Expected ≥2 sub-sectors, got {len(result)}: {result}"
        )

    def test_keyword_coverage(self):
        """每个子方向至少有 3 个关键词"""
        low_kw_subs = []
        for sub_name, keywords in SUB_KEYWORDS.items():
            if len(keywords) < 3:
                low_kw_subs.append((sub_name, len(keywords), keywords))
        assert not low_kw_subs, (
            f"Sub-sectors with <3 keywords: {low_kw_subs}"
        )

    def test_stock_keyword_coverage(self):
        """每个子方向有 stock_keywords"""
        missing = []
        for sub_name in SUB_KEYWORDS:
            if sub_name not in STOCK_TO_SUB_KEYWORDS:
                missing.append(sub_name)
            elif len(STOCK_TO_SUB_KEYWORDS[sub_name]) == 0:
                missing.append(f"{sub_name} (empty)")
        assert not missing, (
            f"Sub-sectors missing stock_keywords: {missing}"
        )

    def test_tushare_mapping(self):
        """TUSHARE_TO_SHENWAN 有 ≥110 个 key"""
        assert len(TUSHARE_TO_SHENWAN) >= 110, (
            f"Expected ≥110 tushare mappings, got {len(TUSHARE_TO_SHENWAN)}"
        )

    def test_industry_keywords(self):
        """INDUSTRY_KEYWORDS 有 29 个 key"""
        assert len(INDUSTRY_KEYWORDS) == 29, (
            f"Expected 29 industry keyword entries, "
            f"got {len(INDUSTRY_KEYWORDS)}"
        )

    def test_valid_industries(self):
        """VALID_INDUSTRIES 和 INDUSTRY_KEYWORDS 一致"""
        expected = set(INDUSTRY_KEYWORDS.keys())
        assert VALID_INDUSTRIES == expected, (
            f"VALID_INDUSTRIES differs from INDUSTRY_KEYWORDS keys.\n"
            f"Only in VALID_INDUSTRIES: {VALID_INDUSTRIES - expected}\n"
            f"Only in INDUSTRY_KEYWORDS: {expected - VALID_INDUSTRIES}"
        )
