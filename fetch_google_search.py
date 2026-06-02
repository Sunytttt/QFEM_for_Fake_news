#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
补全Google搜索数据的脚本
使用Bocha AI搜索API为每个新闻样本获取Google搜索结果

API文档: https://bocha-ai.feishu.cn/wiki/RXEOw02rFiwzGSkd9mUcqoeAnNK
"""

import os
import json
import time
import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# API配置
API_KEY = "sk-a9c4ad4fb5fb427bb06dcf4f771fb8b2"
API_ENDPOINT = "https://api.bocha.cn/v1/web-search"
API_TIMEOUT = 120  # 增加到120秒
REQUEST_TIMEOUT = 3.0  # 请求间隔3秒
MAX_RETRIES = 3  # 最多重试3次

# 来源类型映射 (num_source_types=5)
SOURCE_TYPE_MAPPING = {
    "news": 0,          # 新闻网站
    "blog": 1,          # 博客
    "social": 2,        # 社交媒体/微博
    "forum": 3,         # 论坛
    "other": 4,         # 其他
}

def create_session():
    """创建带重试策略的requests会话"""
    session = requests.Session()

    # 兼容不同版本的urllib3
    try:
        retry_strategy = Retry(
            total=MAX_RETRIES,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],  # 新版本
            backoff_factor=1
        )
    except TypeError:
        # 旧版本使用method_whitelist
        retry_strategy = Retry(
            total=MAX_RETRIES,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["POST"],  # 旧版本
            backoff_factor=1
        )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

def classify_source_type(url: str, site_name: str, title: str) -> int:
    """
    根据URL、网站名称和标题推断来源类型
    返回 0-4 的类型ID
    """
    url_lower = url.lower()
    site_lower = site_name.lower()
    title_lower = title.lower()

    # 新闻网站特征
    news_keywords = [
        "news", "xinhuanet", "people.com", "chinanews", "sina",
        "163", "qq.com", "sohu", "ifeng", "cnr.cn", "cnn", "bbc",
        "reuters", "chinadaily", "cgtn", "cctv", "mtime"
    ]
    if any(kw in url_lower or kw in site_lower for kw in news_keywords):
        return SOURCE_TYPE_MAPPING["news"]

    # 博客特征
    blog_keywords = ["blog", "blogger", "wordpress", "medium", "jianshu"]
    if any(kw in url_lower or kw in site_lower for kw in blog_keywords):
        return SOURCE_TYPE_MAPPING["blog"]

    # 社交媒体/微博特征
    social_keywords = [
        "weibo", "wechat", "douyin", "tiktok", "twitter", "facebook",
        "instagram", "reddit", "youtube", "xiaohongshu", "bilibili",
        "抖音", "微博", "小红书", "B站"
    ]
    if any(kw in url_lower or kw in site_lower for kw in social_keywords):
        return SOURCE_TYPE_MAPPING["social"]

    # 论坛特征
    forum_keywords = ["forum", "tieba", "zhihu", "baidu", "ask", "bbs"]
    if any(kw in url_lower or kw in site_lower for kw in forum_keywords):
        return SOURCE_TYPE_MAPPING["forum"]

    # 默认其他
    return SOURCE_TYPE_MAPPING["other"]

def calculate_time_diff(date_str: Optional[str]) -> float:
    """
    计算发布时间和当前时间的差异（天数）
    支持格式: 2025-02-23T08:18:30Z 或 2025-02-23T08:18:30+08:00
    如果无法解析，返回0
    """
    if not date_str:
        return 0.0

    try:
        # 处理ISO格式时间
        date_str = str(date_str).replace("Z", "+00:00")

        # 尝试解析时间
        if "T" in date_str:
            # ISO格式
            pub_dt = datetime.fromisoformat(date_str.replace("Z", ""))
        else:
            # 简单日期格式
            pub_dt = datetime.strptime(date_str[:10], "%Y-%m-%d")

        # 当前时间（使用发布时间作为参考）
        cur_dt = datetime.now()
        delta = (cur_dt - pub_dt).days
        return max(0.0, float(delta))
    except Exception as e:
        logger.debug(f"Failed to parse date {date_str}: {e}")
        return 0.0

def search_api_call(query: str, max_results: int = 10) -> Dict[str, Any]:
    """
    调用Bocha AI Web Search API获取结果（带重试机制）
    """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "query": query,
        "summary": True,
        "count": min(max_results, 50),
        "freshness": "noLimit"
    }

    session = create_session()

    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.debug(f"API调用 (尝试 {attempt+1}/{MAX_RETRIES+1}): {query[:50]}")

            response = session.post(
                API_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT
            )

            if response.status_code != 200:
                logger.warning(f"API返回非200状态码: {response.status_code}")
                if attempt < MAX_RETRIES:
                    wait_time = (attempt + 1) * 2
                    logger.info(f"等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    return {"results": [], "status": "error"}

            data = response.json()

            # 检查返回状态
            if data.get("code") != 200:
                logger.warning(f"API返回错误代码 {data.get('code')}: {data.get('msg')}")
                if attempt < MAX_RETRIES:
                    wait_time = (attempt + 1) * 2
                    logger.info(f"等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    return {"results": [], "status": "error"}

            # 提取搜索结果
            web_pages = data.get("data", {}).get("webPages", {}).get("value", [])

            results = []
            for item in web_pages:
                results.append({
                    "name": item.get("name", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                    "siteName": item.get("siteName", ""),
                    "dateLastCrawled": item.get("dateLastCrawled") or item.get("datePublished")
                })

            logger.debug(f"API调用成功，返回{len(results)}条结果")
            return {"results": results, "status": "success"}

        except requests.exceptions.Timeout:
            logger.warning(f"API调用超时 (尝试 {attempt+1}/{MAX_RETRIES+1})")
            if attempt < MAX_RETRIES:
                wait_time = (attempt + 1) * 2
                logger.info(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"API调用失败 (尝试{MAX_RETRIES+1}次后放弃): {query[:50]}")

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"连接错误 (尝试 {attempt+1}/{MAX_RETRIES+1}): {e}")
            if attempt < MAX_RETRIES:
                wait_time = (attempt + 1) * 2
                logger.info(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"连接失败 (尝试{MAX_RETRIES+1}次后放弃): {query[:50]}")

        except Exception as e:
            logger.error(f"API调用异常: {e}")
            if attempt < MAX_RETRIES:
                wait_time = (attempt + 1) * 2
                logger.info(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                return {"results": [], "status": "error"}

    return {"results": [], "status": "error"}

def process_search_results(results: List[Dict]) -> tuple:
    """
    处理搜索结果，提取所需信息
    返回: (texts, source_types, time_diffs)
    """
    texts = []
    source_types = []
    time_diffs = []

    for result in results:
        # 提取文本（title + snippet）
        title = result.get("name", "")
        snippet = result.get("snippet", "")
        text = f"{title}。{snippet}" if snippet else title

        if text.strip():
            texts.append(text)

            # 分类来源
            url = result.get("url", "")
            site_name = result.get("siteName", "")
            src_type = classify_source_type(url, site_name, title)
            source_types.append(src_type)

            # 计算时间差
            publish_date = result.get("dateLastCrawled")
            time_diff = calculate_time_diff(publish_date)
            time_diffs.append(time_diff)

    return texts, source_types, time_diffs

def fetch_google_search_for_sample(news_text: str, max_results: int = 10) -> Dict[str, Any]:
    """
    为一个新闻样本获取Google搜索数据
    """
    # 使用新闻标题或前100个字符作为查询
    query = news_text[:100].strip()

    if not query:
        return {
            'google_search_text': [""] * max_results,
            'google_source_type': [4] * max_results,
            'google_time_diff': [0.0] * max_results,
            'success': False
        }

    logger.debug(f"Searching: {query[:50]}...")

    # 调用API
    api_result = search_api_call(query, max_results)

    if api_result.get("status") != "success" or not api_result.get("results"):
        logger.debug(f"No results for query: {query[:50]}")
        return {
            'google_search_text': [""] * max_results,
            'google_source_type': [4] * max_results,
            'google_time_diff': [0.0] * max_results,
            'success': False
        }

    # 处理结果
    texts, source_types, time_diffs = process_search_results(
        api_result.get("results", [])
    )

    # 填充到max_results
    while len(texts) < max_results:
        texts.append("")
        source_types.append(4)
        time_diffs.append(0.0)

    return {
        'google_search_text': texts[:max_results],
        'google_source_type': source_types[:max_results],
        'google_time_diff': time_diffs[:max_results],
        'success': len(texts) > 0 and any(t.strip() for t in texts)
    }

def enrich_dataset(input_path: str, output_path: str, max_samples: Optional[int] = None):
    """
    为整个数据集补全Google搜索数据
    """

    logger.info(f"Loading dataset from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    if max_samples:
        data = data[:max_samples]

    logger.info(f"Processing {len(data)}/{total} samples")

    # 统计
    success_count = 0
    skip_count = 0
    error_count = 0

    # 处理每个样本
    for idx, sample in enumerate(tqdm(data, desc="Enriching")):
        try:
            # 检查是否已有Google数据
            existing_google = sample.get("google_search_text", [])
            if existing_google and len(existing_google) > 0 and any(t.strip() if isinstance(t, str) else False for t in existing_google):
                skip_count += 1
                continue

            # 获取查询文本
            query_text = sample.get("ver_news_text", "")
            if not query_text:
                query_text = sample.get("text", "")
            if not query_text:
                logger.debug(f"Sample {idx} has no text")
                error_count += 1
                continue

            # 获取搜索结果
            search_result = fetch_google_search_for_sample(query_text, max_results=10)

            # 添加到样本
            sample['google_search_text'] = search_result['google_search_text']
            sample['google_source_type'] = search_result['google_source_type']
            sample['google_time_diff'] = search_result['google_time_diff']

            if search_result['success']:
                success_count += 1

            # 速率控制
            time.sleep(REQUEST_TIMEOUT)

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            error_count += 1
            # 设置默认值
            sample['google_search_text'] = [""] * 10
            sample['google_source_type'] = [4] * 10
            sample['google_time_diff'] = [0.0] * 10

    # 保存结果
    logger.info(f"Saving enriched dataset to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Complete!")
    logger.info(f"  - Total processed: {len(data)}")
    logger.info(f"  - Successful: {success_count}")
    logger.info(f"  - Skipped: {skip_count}")
    logger.info(f"  - Errors: {error_count}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="补全Google搜索数据")
    parser.add_argument("--input", required=True, help="输入JSON文件路径")
    parser.add_argument("--output", required=True, help="输出JSON文件路径")
    parser.add_argument("--max-samples", type=int, default=None, help="最多处理的样本数（用于测试）")

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("Bocha AI Google搜索数据补全工具 (带重试机制)")
    logger.info("="*70)
    logger.info(f"API Key: {API_KEY[:10]}...")
    logger.info(f"API Endpoint: {API_ENDPOINT}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Timeout: {API_TIMEOUT}秒")
    logger.info(f"Max Retries: {MAX_RETRIES}次")
    if args.max_samples:
        logger.info(f"Max samples: {args.max_samples} (test mode)")
    logger.info("="*70)

    enrich_dataset(args.input, args.output, max_samples=args.max_samples)

if __name__ == "__main__":
    main()
