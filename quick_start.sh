#!/bin/bash
# 快速启动脚本 - 包含诊断 + 训练

source /data/SYT/myenv/bin/activate
cd /home/sunyuantao/icassp

echo "🚀 启动 CAMER+CEIN 训练（含自动诊断）"
echo ""

# 直接运行已修改的训练脚本
./run_train.sh
