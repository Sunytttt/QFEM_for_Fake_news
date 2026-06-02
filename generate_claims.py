#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补全 claims 字段 —— 调用 SiliconFlow LLM API 将新闻分解为原子事实声明
用法:
    source /data/SYT/myenv/bin/activate
    python generate_claims.py --input data/group_data_standardized/group3.csv.json
"""

import os
import json
import time
import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ---- 配置 ----
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-ozezlbfcrwsiupjrrgmwwnkimosscyoxhiglsymeosttajfw"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_CLAIMS = 10

SYSTEM_PROMPT = (
    "你是一个事实核查助手。给定一条新闻，将其分解为独立的、可验证的原子事实声明。"
    "每个声明应是一个简短的陈述句，可以独立判断真假。"
    f"输出JSON数组格式，如[\"声明1\",\"声明2\",...]。只输出JSON数组，不要其他文字。最多输出{MAX_CLAIMS}个声明。"
    "如果新闻内容太短或无法分解，输出至少1个声明。"
)


def call_llm(text: str, max_retries: int = 3) -> list:
    """调用 LLM 分解新闻为原子声明，带重试。"""
    # 截断过长文本
    text = text[:800]

    for attempt in range(max_retries):
        try:
            # 请求前短暂等待，避免触发限流
            time.sleep(0.5)

            resp = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"请将以下新闻分解为原子事实声明：\n\n{text}"},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.3,
                },
                timeout=60,
            )

            if resp.status_code == 429:
                # 限流，指数退避
                wait = 10 * (attempt + 1)
                logging.warning("Rate limited, waiting %ds...", wait)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # 尝试直接解析 JSON
            try:
                claims = json.loads(content)
                if isinstance(claims, list):
                    return [str(c) for c in claims[:MAX_CLAIMS]]
            except json.JSONDecodeError:
                pass

            # 尝试从文本中提取 JSON 数组
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                try:
                    claims = json.loads(match.group())
                    if isinstance(claims, list):
                        return [str(c) for c in claims[:MAX_CLAIMS]]
                except json.JSONDecodeError:
                    pass

            # 按行分割作为 fallback
            lines = [l.strip().lstrip("0123456789.-) ") for l in content.split("\n") if l.strip()]
            if lines:
                return lines[:MAX_CLAIMS]

            return [text[:100]]

        except requests.exceptions.RequestException as e:
            logging.warning("API error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            time.sleep(3 * (attempt + 1))

    # 全部重试失败
    return [text[:100]]


def process_record(idx: int, record: dict) -> tuple:
    """处理单条记录，返回 (idx, claims)。"""
    text = record.get("ver_news_text", "")
    if not text.strip():
        return idx, [""]
    claims = call_llm(text)
    return idx, claims


def main():
    parser = argparse.ArgumentParser(description="为数据集补全 claims 字段")
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output", default=None, help="输出路径（默认覆盖原文件）")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数")
    parser.add_argument("--start", type=int, default=0, help="从第几条开始（用于断点续跑）")
    parser.add_argument("--limit", type=int, default=0, help="只处理前N条（0=全部）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output_path = args.output or args.input

    # 加载数据
    logging.info("Loading %s ...", args.input)
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    logging.info("Loaded %d records", len(data))

    # 确定处理范围
    end = len(data) if args.limit == 0 else min(args.start + args.limit, len(data))
    to_process = []
    for i in range(args.start, end):
        # 跳过已有 claims 的记录
        existing = data[i].get("claims")
        if existing and len(existing) > 0 and any(c.strip() for c in existing):
            continue
        to_process.append(i)

    logging.info("Need to process: %d records (skip %d already done)",
                 len(to_process), (end - args.start) - len(to_process))

    if not to_process:
        logging.info("Nothing to do, all records already have claims.")
        return

    # 并发处理
    done = 0
    total = len(to_process)
    save_interval = 200  # 每处理200条保存一次

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_record, idx, data[idx]): idx
            for idx in to_process
        }

        for future in as_completed(futures):
            idx, claims = future.result()
            data[idx]["claims"] = claims
            done += 1

            if done % 50 == 0:
                logging.info("Progress: %d/%d (%.1f%%)", done, total, 100 * done / total)

            # 定期保存
            if done % save_interval == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logging.info("Checkpoint saved (%d/%d)", done, total)

    # 最终保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 统计
    has_claims = sum(1 for r in data if r.get("claims") and any(c.strip() for c in r["claims"]))
    logging.info("Done! %d/%d records now have claims. Saved to %s",
                 has_claims, len(data), output_path)


if __name__ == "__main__":
    main()
