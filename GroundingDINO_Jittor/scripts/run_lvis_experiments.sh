#!/bin/bash
# ============================================================
# Grounding DINO Jittor - LVIS 实验脚本
# 
# 在 Linux 服务器上运行 Zero-Shot 测试和微调实验
#
# 用法:
#   chmod +x scripts/run_lvis_experiments.sh
#   ./scripts/run_lvis_experiments.sh
# ============================================================

set -e  # 遇到错误立即退出

# ============================================================
# 配置 - 请根据实际情况修改
# ============================================================

# 数据路径
LVIS_TRAIN_ANN="/path/to/lvis/lvis_v1_train.json"
LVIS_VAL_ANN="/path/to/lvis/lvis_v1_val.json"
IMAGE_DIR="/path/to/coco"  # COCO 图像目录 (包含 train2017, val2017)

# 权重路径
PRETRAINED_WEIGHTS="weights/groundingdino_swint_ogc_jittor.pkl"
PYTORCH_WEIGHTS="weights/groundingdino_swint_ogc.pth"

# 输出目录
OUTPUT_BASE="outputs"
ZEROSHOT_OUTPUT="${OUTPUT_BASE}/lvis_zeroshot"
FINETUNE_OUTPUT="${OUTPUT_BASE}/lvis_finetune"

# 训练参数
EPOCHS=12
BATCH_SIZE=4
LR=1e-4
NUM_WORKERS=8

# ============================================================
# 环境检查
# ============================================================

echo "============================================================"
echo "Grounding DINO Jittor - LVIS Experiments"
echo "============================================================"
echo "Time: $(date)"
echo ""

# 检查 Python 和 Jittor
echo "Checking environment..."
python -c "import jittor as jt; print(f'Jittor version: {jt.__version__}')"
python -c "import jittor as jt; print(f'CUDA available: {jt.has_cuda}')"

# 检查数据路径
echo ""
echo "Checking data paths..."

if [ ! -f "$LVIS_TRAIN_ANN" ]; then
    echo "Warning: LVIS train annotation not found: $LVIS_TRAIN_ANN"
    echo "Please update the path in this script"
fi

if [ ! -f "$LVIS_VAL_ANN" ]; then
    echo "Warning: LVIS val annotation not found: $LVIS_VAL_ANN"
    echo "Please update the path in this script"
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Warning: Image directory not found: $IMAGE_DIR"
    echo "Please update the path in this script"
fi

# ============================================================
# Step 1: 准备权重
# ============================================================

echo ""
echo "============================================================"
echo "Step 1: Preparing weights"
echo "============================================================"

mkdir -p weights

# 下载 PyTorch 权重 (如果不存在)
if [ ! -f "$PYTORCH_WEIGHTS" ]; then
    echo "Downloading pretrained weights..."
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O "$PYTORCH_WEIGHTS"
    echo "Downloaded to $PYTORCH_WEIGHTS"
else
    echo "PyTorch weights already exist: $PYTORCH_WEIGHTS"
fi

# 转换权重 (如果不存在)
if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
    echo "Converting weights to Jittor format..."
    python scripts/convert_weights_pytorch_to_jittor.py \
        --pytorch_weight "$PYTORCH_WEIGHTS" \
        --output "$PRETRAINED_WEIGHTS"
    echo "Converted to $PRETRAINED_WEIGHTS"
else
    echo "Jittor weights already exist: $PRETRAINED_WEIGHTS"
fi

# ============================================================
# Step 2: Zero-Shot 评估
# ============================================================

echo ""
echo "============================================================"
echo "Step 2: Zero-Shot Evaluation on LVIS"
echo "============================================================"

mkdir -p "$ZEROSHOT_OUTPUT"

python scripts/eval_lvis_zeroshot.py \
    --checkpoint "$PRETRAINED_WEIGHTS" \
    --lvis_ann "$LVIS_VAL_ANN" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$ZEROSHOT_OUTPUT" \
    --box_threshold 0.25 \
    --nms_threshold 0.5 \
    --use_gpu

echo "Zero-shot results saved to $ZEROSHOT_OUTPUT"

# ============================================================
# Step 3: 微调
# ============================================================

echo ""
echo "============================================================"
echo "Step 3: Fine-tuning on LVIS"
echo "============================================================"

mkdir -p "$FINETUNE_OUTPUT"

python scripts/finetune_lvis.py \
    --checkpoint "$PRETRAINED_WEIGHTS" \
    --lvis_train "$LVIS_TRAIN_ANN" \
    --lvis_val "$LVIS_VAL_ANN" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$FINETUNE_OUTPUT" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --num_workers $NUM_WORKERS \
    --freeze_backbone \
    --freeze_text_encoder \
    --log_interval 100 \
    --save_interval 2 \
    --eval_interval 1 \
    --use_gpu

echo "Fine-tuning completed. Models saved to $FINETUNE_OUTPUT"

# ============================================================
# Step 4: 微调后评估
# ============================================================

echo ""
echo "============================================================"
echo "Step 4: Evaluation after Fine-tuning"
echo "============================================================"

FINETUNED_OUTPUT="${OUTPUT_BASE}/lvis_finetuned_eval"
mkdir -p "$FINETUNED_OUTPUT"

python scripts/eval_lvis_zeroshot.py \
    --checkpoint "${FINETUNE_OUTPUT}/best_model.pkl" \
    --lvis_ann "$LVIS_VAL_ANN" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$FINETUNED_OUTPUT" \
    --box_threshold 0.25 \
    --nms_threshold 0.5 \
    --use_gpu

echo "Fine-tuned model results saved to $FINETUNED_OUTPUT"

# ============================================================
# 结果汇总
# ============================================================

echo ""
echo "============================================================"
echo "Experiment Summary"
echo "============================================================"

echo ""
echo "Zero-Shot Results:"
if [ -f "${ZEROSHOT_OUTPUT}/results.json" ]; then
    cat "${ZEROSHOT_OUTPUT}/results.json"
fi

echo ""
echo "Fine-tuned Results:"
if [ -f "${FINETUNED_OUTPUT}/results.json" ]; then
    cat "${FINETUNED_OUTPUT}/results.json"
fi

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "============================================================"
echo "Outputs:"
echo "  - Zero-shot: $ZEROSHOT_OUTPUT"
echo "  - Fine-tuned model: $FINETUNE_OUTPUT"
echo "  - Fine-tuned eval: $FINETUNED_OUTPUT"
echo ""
echo "Time: $(date)"

