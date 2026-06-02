#!/bin/bash
# =============================================================================
# CAMER+CEIN 跨源假新闻检测 — 训练 + 跨域测试脚本
# 训练集: group3  |  跨域测试: group1, group2
# =============================================================================

# ---- 环境 ----
source /data/SYT/myenv/bin/activate
cd /home/sunyuantao/icassp

# 禁止联网下载，强制使用本地缓存的模型
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# CUDA优化设置
# export CUDA_LAUNCH_BLOCKING=1  # 仅调试时启用，训练时关闭以提升性能

# ---- 参数（可直接修改） ----
DATA_DIR="data/group_data_standardized"
TRAIN_DATA="${DATA_DIR}/group3.csv.json"
BERT_MODEL="xlm-roberta-large"
OUTPUT_DIR="outputs_camer_v2"
LOG_FILE="train_log_$(date +%Y%m%d_%H%M%S).txt"

BATCH_SIZE=1          # 每卡batch_size (4卡×1=4, 累积8步后有效batch=32)
GRADIENT_ACCUMULATION=8  # 梯度累积步数，有效batch_size = 1×4×8 = 32
EPOCHS=10
LR=1e-5
MAX_LENGTH=192        # 文本序列最大长度 (平衡性能和显存)
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
VAL_RATIO=0.1
TEST_RATIO=0.1
MAX_SIM=8             # 相似新闻数量
MAX_GOOGLE=8          # 搜索结果数量
MAX_CLAIMS=8          # claims数量
MAX_COMMENTS=10       # 评论数量
IMAGE_SIZE=224
SEED=42
NUM_WORKERS=4         # 每卡workers数量
CLAIM_LOSS_WEIGHT=0.3
TEMPERATURE=1.0
EVIDENCE_DROPOUT=0.15  # 训练时随机丢弃证据，增强对稀疏google证据的鲁棒性
NPROC_PER_NODE=4      # 使用4张GPU
FP16=true             # 启用混合精度训练节省显存

# ---- 诊断CUDA数值稳定性 ----
echo "============================================="
echo " Running CUDA Numerical Stability Diagnostics"
echo "============================================="
python3 diagnose_cuda_issue.py
DIAG_EXIT=$?
if [ $DIAG_EXIT -ne 0 ]; then
    echo "⚠️  Warning: Diagnostics encountered issues"
    echo "Continuing with training anyway..."
fi
echo ""

# ---- 运行 ----
echo "============================================="
echo " CAMER+CEIN Cross-Source Fake News Detection"
echo "============================================="
echo "Train data : ${TRAIN_DATA}"
echo "Output dir : ${OUTPUT_DIR}"
echo "Log file   : ${LOG_FILE}"
echo "Batch size : ${BATCH_SIZE}"
echo "Epochs     : ${EPOCHS}"
echo "LR         : ${LR}"
echo "Max length : ${MAX_LENGTH}"
echo "============================================="
echo ""

torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=29500 train_multimodal.py \
    --data_path "${TRAIN_DATA}" \
    --bert_model_name "${BERT_MODEL}" \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    $([ "$FP16" = "true" ] && echo "--fp16") \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --max_length ${MAX_LENGTH} \
    --warmup_ratio ${WARMUP_RATIO} \
    --weight_decay ${WEIGHT_DECAY} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --val_ratio ${VAL_RATIO} \
    --test_ratio ${TEST_RATIO} \
    --max_sim ${MAX_SIM} \
    --max_google ${MAX_GOOGLE} \
    --max_claims ${MAX_CLAIMS} \
    --max_comments ${MAX_COMMENTS} \
    --image_size ${IMAGE_SIZE} \
    --seed ${SEED} \
    --output_dir "${OUTPUT_DIR}" \
    --save_best \
    --num_workers ${NUM_WORKERS} \
    --claim_loss_weight ${CLAIM_LOSS_WEIGHT} \
    --temperature ${TEMPERATURE} \
    --evidence_dropout ${EVIDENCE_DROPOUT} \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "============================================="
echo " Done. Log saved to: ${LOG_FILE}"
echo "============================================="
