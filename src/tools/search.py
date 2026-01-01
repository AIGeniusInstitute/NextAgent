"""
搜索工具
========

提供网络搜索功能（模拟实现）。
"""

import json
import random
from typing import List, Optional
from datetime import datetime

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchInput(BaseModel):
    """网络搜索输入参数"""
    query: str = Field(
        description="搜索查询关键词"
    )
    num_results: Optional[int] = Field(
        default=5,
        description="返回结果数量",
        ge=1,
        le=10
    )


class SearchResult(BaseModel):
    """搜索结果模型"""
    title: str
    url: str
    snippet: str
    source: str


class WebSearch:
    """
    网络搜索器
    
    当前为模拟实现，返回预设的搜索结果。
    生产环境应对接真实的搜索 API（如 Google、Bing、Serper 等）。
    
    扩展方式：
    1. 继承此类并重写 search 方法
    2. 或配置 API 密钥连接真实服务
    """
    
    # 模拟搜索结果库
    MOCK_RESULTS = {
        "python": [
            SearchResult(
                title="Python 官方文档",
                url="https://docs.python.org/",
                snippet="Python 是一种易于学习、功能强大的编程语言。它具有高效的高级数据结构和简单而有效的面向对象编程方法。",
                source="python.org"
            ),
            SearchResult(
                title="Python 教程 - 菜鸟教程",
                url="https://www.runoob.com/python/",
                snippet="Python 是一个高层次的结合了解释性、编译性、互动性和面向对象的脚本语言。",
                source="runoob.com"
            ),
            SearchResult(
                title="Python 入门指南",
                url="https://docs.python.org/zh-cn/3/tutorial/",
                snippet="本教程非正式地介绍 Python 语言和系统的基本概念和功能。",
                source="python.org"
            ),
        ],
        "爬虫": [
            SearchResult(
                title="Python 爬虫教程",
                url="https://example.com/crawler-tutorial",
                snippet="学习使用 Python 进行网页爬取，包括 requests、BeautifulSoup、Scrapy 等常用库的使用方法。",
                source="example.com"
            ),
            SearchResult(
                title="requests 库文档",
                url="https://docs.python-requests.org/",
                snippet="Requests 是一个简洁且简单的 Python HTTP 库，让 HTTP 请求变得更加人性化。",
                source="python-requests.org"
            ),
            SearchResult(
                title="BeautifulSoup 教程",
                url="https://www.crummy.com/software/BeautifulSoup/",
                snippet="Beautiful Soup 是一个用于解析 HTML 和 XML 文档的 Python 库。",
                source="crummy.com"
            ),
        ],
        "hacker news": [
            SearchResult(
                title="Hacker News",
                url="https://news.ycombinator.com/",
                snippet="Hacker News 是一个社会化新闻网站，专注于计算机科学和创业。由 Y Combinator 创办和运营。",
                source="ycombinator.com"
            ),
            SearchResult(
                title="Hacker News API",
                url="https://github.com/HackerNews/API",
                snippet="The official Hacker News API. Documentation and examples for accessing HN data.",
                source="github.com"
            ),
            SearchResult(
                title="HN 爬取指南",
                url="https://example.com/hn-scraping",
                snippet="如何爬取 Hacker News 网站，包括文章标题、链接、评分等信息。",
                source="example.com"
            ),
        ],
        "default": [
            SearchResult(
                title="搜索结果 1",
                url="https://example.com/result1",
                snippet="这是一个模拟的搜索结果。在实际应用中，这里会显示真实的搜索结果内容。",
                source="example.com"
            ),
            SearchResult(
                title="搜索结果 2",
                url="https://example.com/result2",
                snippet="另一个模拟搜索结果。系统当前使用模拟数据，生产环境需要对接真实搜索API。",
                source="example.com"
            ),
        ],
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化搜索器
        
        Args:
            api_key: 搜索 API 密钥（可选）
        """
        self.api_key = api_key
        self.logger = get_logger(self.__class__.__name__)
    
    def search(
        self,
        query: str,
        num_results: int = 5
    ) -> List[SearchResult]:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        self.logger.info(f"搜索: {query}")
        
        # 模拟搜索延迟
        import time
        time.sleep(0.5)
        
        # 匹配模拟结果
        results = []
        query_lower = query.lower()
        
        for keyword, mock_results in self.MOCK_RESULTS.items():
            if keyword != "default" and keyword in query_lower:
                results.extend(mock_results)
        
        # 如果没有匹配，使用默认结果
        if not results:
            results = self.MOCK_RESULTS["default"].copy()
            # 动态生成一些结果
            results.append(SearchResult(
                title=f"关于 '{query}' 的搜索结果",
                url=f"https://example.com/search?q={query}",
                snippet=f"这是关于 '{query}' 的搜索结果。包含相关信息和资源链接。",
                source="example.com"
            ))
        
        # 打乱顺序增加真实感
        random.shuffle(results)
        
        # 限制数量
        results = results[:num_results]
        
        self.logger.info(f"返回 {len(results)} 条结果")
        
        return results
    
    def format_results(self, results: List[SearchResult]) -> str:
        """
        格式化搜索结果为可读字符串
        
        Args:
            results: 搜索结果列表
            
        Returns:
            格式化的字符串
        """
        if not results:
            return "未找到相关结果"
        
        lines = [f"找到 {len(results)} 条结果:\n"]
        
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. **{result.title}**")
            lines.append(f"   链接: {result.url}")
            lines.append(f"   摘要: {result.snippet}")
            lines.append(f"   来源: {result.source}")
            lines.append("")
        
        return "\n".join(lines)


# 创建工具实例
_web_search = WebSearch()


@tool(args_schema=WebSearchInput)
def web_search_tool(query: str, num_results: int = 5) -> str:
    """
    搜索互联网获取信息。
    
    当需要获取最新信息、查找资料或验证事实时使用此工具。
    
    注意：当前为模拟实现，返回预设结果。
    
    使用示例：
    - 搜索 Python 教程: query="Python 入门教程"
    - 查找 API 文档: query="requests 库文档"
    """
    try:
        results = _web_search.search(query, num_results)
        return _web_search.format_results(results)
    except Exception as e:
        logger.error(f"搜索异常: {e}", exc_info=True)
        return f"搜索失败: {str(e)}"