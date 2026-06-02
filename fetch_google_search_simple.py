#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版数据补全脚本 - 基于成功的API调用方式
"""

import os
import json
import time
import requests
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# API配置
API_KEY = "sk-a9c4ad4fb5fb427bb06dcf4f771fb8b2"
API_ENDPOINT = "https://api.bocha.cn/v1/web-search"
API_TIMEOUT = 300  # 300秒超时（5分钟）
REQUEST_TIMEOUT = 5.0  # 请求间隔
MAX_RETRIES = 2  # 简单重试2���

# 来源类型映射
SOURCE_TYPE_MAPPING = {"news": 0, "blog": 1, "social": 2, "forum": 3, "other": 4}

def classify_source_type(url: str, site_name: str) -> int:
    """简单的来源分类"""
    url_lower = url.lower()
    site_lower = site_name.lower()

    news_keywords = ["news", "xinhuanet", "people.com", "sina", "sohu", "ifeng"]
    blog_keywords = ["blog", "wordpress", "medium"]
    social_keywords = ["weibo", "douyin", "tiktok", "twitter", "youtube"]
    forum_keywords = ["forum", "tieba", "zhihu", "baidu"]

    if any(kw in url_lower or kw in site_lower for kw in news_keywords):
        return 0
    if any(kw in url_lower or kw in site_lower for kw in blog_keywords):
        return 1
    if any(kw in url_lower or kw in site_lower for kw in social_keywords):
        return 2
    if any(kw in url_lower or kw in site_lower for kw in forum_keywords):
        return 3
    return 4

def simple_search(query: str, max_results: int = 10) -> Dict[str, Any]:
    """简化搜索 - 增加超时和基础重试"""

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

    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.debug(f"API调用尝试 {attempt+1}/{MAX_RETRIES+1}: {query[:40]}")
            response = requests.post(
                API_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT
            )

            if response.status_code != 200:
                logger.warning(f"HTTP错误: {response.status_code}")
                if attempt < MAX_RETRIES:
                    wait_time = (attempt + 1) * 3
                    logger.info(f"等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                return {"results": [], "status": "error"}

            data = response.json()

            if data.get("code") != 200:
                logger.warning(f"API错误: {data.get('msg')}")
                if attempt < MAX_RETRIES:
                    wait_time = (attempt + 1) * 3
                    logger.info(f"等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                return {"results": [], "status": "error"}

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
            logger.warning(f"API超时 (尝试 {attempt+1}/{MAX_RETRIES+1})")
            if attempt < MAX_RETRIES:
                wait_time = (attempt + 1) * 3
                logger.info(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"API调用失败 (尝试{MAX_RETRIES+1}次): {query[:40]}")
                return {"results": [], "status": "error"}
        except Exception as e:
            logger.error(f"API调用异常: {e}")
            if attempt < MAX_RETRIES:
                wait_time = (attempt + 1) * 3
                logger.info(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                return {"results": [], "status": "error"}

    return {"results": [], "status": "error"}

def fetch_search_data(news_text: str, max_results: int = 10) -> Dict[str, Any]:
    """获取一个样本的Google搜索数据"""

    query = news_text[:100].strip()
    if not query:
        return {
            'google_search_text': [""] * max_results,
            'google_source_type': [4] * max_results,
            'google_time_diff': [0.0] * max_results,
            'success': False
        }

    api_result = simple_search(query, max_results)

    if api_result.get("status") != "success" or not api_result.get("results"):
        return {
            'google_search_text': [""] * max_results,
            'google_source_type': [4] * max_results,
            'google_time_diff': [0.0] * max_results,
            'success': False
        }

    texts = []
    source_types = []
    time_diffs = []

    for result in api_result.get("results", []):
        title = result.get("name", "")
        snippet = result.get("snippet", "")
        text = f"{title}。{snippet}" if snippet else title

        if text.strip():
            texts.append(text)
            src_type = classify_source_type(result.get("url", ""), result.get("siteName", ""))
            source_types.append(src_type)
            time_diffs.append(0.0)

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
    """补全数据集"""

    logger.info(f"加载数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    if max_samples:
        data = data[:max_samples]

    logger.info(f"处理样本: {len(data)}/{total}")

    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, sample in enumerate(tqdm(data, desc="处理中")):
        try:
            # 检查是否已有数据
            existing = sample.get("google_search_text", [])
            if existing and len(existing) > 0 and any(t.strip() if isinstance(t, str) else False for t in existing):
                skip_count += 1
                continue

            # 获取查询文本
            query_text = sample.get("ver_news_text", "") or sample.get("text", "")
            if not query_text:
                error_count += 1
                continue

            # 搜索
            result = fetch_search_data(query_text, max_results=10)

            # 更新样本
            sample['google_search_text'] = result['google_search_text']
            sample['google_source_type'] = result['google_source_type']
            sample['google_time_diff'] = result['google_time_diff']

            if result['success']:
                success_count += 1

            # 延迟
            time.sleep(REQUEST_TIMEOUT)

        except Exception as e:
            logger.error(f"样本{idx}处理失败: {e}")
            error_count += 1
            sample['google_search_text'] = [""] * 10
            sample['google_source_type'] = [4] * 10
            sample['google_time_diff'] = [0.0] * 10

    # 保存
    logger.info(f"保存数据: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 完成！")
    logger.info(f"  总数: {len(data)}")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  跳过: {skip_count}")
    logger.info(f"  错误: {error_count}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="补全Google搜索数据")
    parser.add_argument("--input", required=True, help="输入JSON文件")
    parser.add_argument("--output", required=True, help="输出JSON文件")
    parser.add_argument("--max-samples", type=int, default=None, help="最多处理样本数")

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("数据补全工具 (简化版)")
    logger.info("=" * 70)
    logger.info(f"输入: {args.input}")
    logger.info(f"输出: {args.output}")
    logger.info(f"超时: {API_TIMEOUT}秒")
    logger.info(f"延迟: {REQUEST_TIMEOUT}秒")
    logger.info("=" * 70)

    enrich_dataset(args.input, args.output, max_samples=args.max_samples)

if __name__ == "__main__":
    main()
