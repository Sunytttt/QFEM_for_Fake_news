#!/bin/bash
# =============================================================================
# 补全 claims 字段 — 三个数据集全部处理
# =============================================================================

source /data/SYT/myenv/bin/activate
cd /home/sunyuantao/icassp

DATA_DIR="data/group_data_standardized"
WORKERS=8   # 并发线程数，可根据API限流调整

echo "============================================="
echo "  Generating claims for all datasets"
echo "  Model: Qwen/Qwen2.5-7B-Instruct"
echo "  Workers: ${WORKERS}"
echo "============================================="

for group in group3.csv.json group1.csv.json group2.csv.json; do
    echo ""
    echo ">>> Processing ${group} ..."
    python generate_claims.py \
        --input "${DATA_DIR}/${group}" \
        --workers ${WORKERS} \
        2>&1 | tee "claims_log_${group%.csv.json}.txt"
    echo ">>> ${group} done."
    echo ""
done

echo "============================================="
echo "  All datasets processed!"
echo "============================================="
