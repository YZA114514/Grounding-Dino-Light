#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grounding DINO Jittor - LVIS 微调脚本

在 LVIS 数据集上微调 Grounding DINO 模型。

用法:
    python scripts/finetune_lvis.py \
        --pretrained weights/groundingdino_swint_ogc_jittor.pkl \
        --data_path data/lvis \
        --output_dir outputs/finetune_lvis \
        --epochs 20 \
        --batch_size 2 \
        --lr 1e-4
"""

import os
import sys
import argparse
import pickle
from datetime import datetime

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np

try:
    import jittor as jt
    from jittor import nn
except ImportError:
    print("错误: 请安装 Jittor: pip install jittor")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Grounding DINO LVIS Finetuning")
    
    # 权重
    parser.add_argument("--pretrained", type=str, required=True,
                        help="预训练权重路径 (.pkl)")
    
    # 数据
    parser.add_argument("--data_path", type=str, default="data/lvis",
                        help="LVIS 数据集路径")
    parser.add_argument("--image_dir", type=str, default="data/coco",
                        help="COCO 图像目录")
    
    # 输出
    parser.add_argument("--output_dir", type=str, default="outputs/finetune_lvis",
                        help="输出目录")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=20,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--lr_backbone", type=float, default=1e-5,
                        help="Backbone 学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--lr_drop", type=int, default=15,
                        help="学习率衰减的 epoch")
    
    # 冻结设置
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="冻结 backbone")
    parser.add_argument("--freeze_text_encoder", action="store_true",
                        help="冻结文本编码器")
    
    # 其他
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="保存间隔")
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="评估间隔")
    parser.add_argument("--use_gpu", action="store_true",
                        help="使用 GPU")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    return parser.parse_args()


def freeze_module(module: nn.Module, freeze: bool = True):
    """冻结或解冻模块"""
    for param in module.parameters():
        if freeze:
            param.stop_grad()
        else:
            param.start_grad()


def count_parameters(model: nn.Module):
    """统计参数数量"""
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def load_pretrained_weights(model, weight_path: str, freeze_backbone: bool, freeze_text_encoder: bool):
    """加载预训练权重"""
    print(f"\n加载预训练权重: {weight_path}")
    
    # 加载权重
    with open(weight_path, 'rb') as f:
        pretrained = pickle.load(f)
    
    # 分离 BERT 权重和其他权重
    bert_weights = {}
    other_weights = {}
    for k, v in pretrained.items():
        if k.startswith('bert.'):
            bert_weights[k] = v
        else:
            other_weights[k] = v
    
    print(f"  预训练权重: BERT {len(bert_weights)}, 其他 {len(other_weights)}")
    
    # 加载非 BERT 权重
    model_state = model.state_dict()
    loaded = 0
    
    # 权重名称映射
    name_map = {
        "backbone.0.": "backbone.",
        "transformer.level_embed": "level_embed",
        "transformer.tgt_embed.weight": "tgt_embed.weight",
        "transformer.enc_output.": "enc_output.",
    }
    
    for i in range(6):
        name_map[f"bbox_embed.{i}."] = f"transformer.decoder.bbox_embed.{i}."
    
    for k, v in other_weights.items():
        new_k = k
        for old_prefix, new_prefix in name_map.items():
            if new_k.startswith(old_prefix):
                if new_prefix.endswith('.'):
                    new_k = new_prefix + new_k[len(old_prefix):]
                else:
                    new_k = new_prefix
                break
        
        if new_k in model_state:
            if model_state[new_k].shape == tuple(v.shape):
                model_state[new_k] = jt.array(v)
                loaded += 1
    
    model.load_state_dict(model_state)
    print(f"  加载非 BERT 权重: {loaded}/{len(other_weights)}")
    
    # 加载 BERT 权重
    if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'bert'):
        bert_state = model.text_encoder.bert.state_dict()
        bert_loaded = 0
        
        for k, v in bert_weights.items():
            bert_key = k[5:] if k.startswith('bert.') else k
            if bert_key in bert_state:
                if bert_state[bert_key].shape == tuple(v.shape):
                    bert_state[bert_key] = jt.array(v)
                    bert_loaded += 1
        
        model.text_encoder.bert.load_state_dict(bert_state)
        print(f"  加载 BERT 权重: {bert_loaded}/{len(bert_weights)}")
    
    # 冻结设置
    if freeze_backbone and hasattr(model, 'backbone'):
        freeze_module(model.backbone, True)
        print("  已冻结 backbone")
    
    if freeze_text_encoder and hasattr(model, 'text_encoder'):
        freeze_module(model.text_encoder, True)
        print("  已冻结 text_encoder")
    
    total, trainable = count_parameters(model)
    print(f"\n参数统计:")
    print(f"  总参数: {total:,} ({total/1e6:.2f}M)")
    print(f"  可训练: {trainable:,} ({trainable/1e6:.2f}M)")
    
    return model


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Grounding DINO LVIS 微调")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置 CUDA
    if args.use_gpu:
        jt.flags.use_cuda = 1
        print(f"使用 GPU: {jt.has_cuda}")
    
    # 设置随机种子
    np.random.seed(args.seed)
    jt.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 导入模型和相关模块
    from jittor_implementation.models.groundingdino import GroundingDINO
    from jittor_implementation.models.backbone import SwinTransformer
    from jittor_implementation.models.text_encoder import BERTWrapper
    from jittor_implementation.data import build_lvis_dataset, get_dataloader
    from jittor_implementation.losses import SetCriterion
    from jittor_implementation.train.utils import (
        save_model, load_model, MetricLogger, SmoothedValue, create_logger
    )
    
    # 创建 backbone
    print("\n创建模型...")
    backbone = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        use_checkpoint=False
    )
    
    # 创建主模型
    model = GroundingDINO(
        backbone=backbone,
        num_queries=900,
        hidden_dim=256,
        num_feature_levels=4,
        nheads=8,
        max_text_len=256,
        two_stage_type="standard",
        dec_pred_bbox_embed_share=False,
        two_stage_bbox_embed_share=False,
    )
    
    # 加载预训练权重
    model = load_pretrained_weights(
        model,
        args.pretrained,
        freeze_backbone=args.freeze_backbone,
        freeze_text_encoder=args.freeze_text_encoder
    )
    
    # 创建文本编码器
    bert_model_path = os.path.join(project_root, "models", "bert-base-uncased")
    text_encoder = BERTWrapper(model_name=bert_model_path, max_text_len=256)
    
    # 创建损失函数
    criterion = SetCriterion(
        num_classes=1203,  # LVIS 类别数
        weight_dict={
            "loss_ce": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0
        },
        losses=["labels", "boxes", "giou"],
        eos_coef=0.1
    )
    
    # 创建数据集
    print("\n加载数据集...")
    train_dataset = build_lvis_dataset(
        anno_path=os.path.join(args.data_path, "lvis_v1_train.json"),
        image_dir=os.path.join(args.image_dir, "train2017"),
        is_train=True
    )
    
    val_dataset = build_lvis_dataset(
        anno_path=os.path.join(args.data_path, "lvis_v1_val.json"),
        image_dir=os.path.join(args.image_dir, "val2017"),
        is_train=False
    )
    
    print(f"  训练集: {len(train_dataset)} 张图像")
    print(f"  验证集: {len(val_dataset)} 张图像")
    
    # 创建数据加载器
    train_loader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 创建优化器（只优化可训练参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = jt.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = jt.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.lr_drop],
        gamma=0.1
    )
    
    # 创建日志
    logger = create_logger(args.output_dir, "finetune")
    
    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # 训练
        model.train()
        text_encoder.train()
        
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        
        batch_idx = 0
        for samples, targets in train_loader:
            # 获取文本
            texts = [t.get('text', '') for t in targets]
            
            # 文本编码
            text_dict = text_encoder(texts)
            
            # 前向传播
            outputs = model(samples, text_dict)
            
            # 计算损失
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] * criterion.weight_dict.get(k, 1.0) 
                        for k in loss_dict.keys())
            
            # 反向传播
            optimizer.step(losses)
            
            # 记录
            loss_values = {k: v.item() if isinstance(v, jt.Var) else v 
                          for k, v in loss_dict.items()}
            metric_logger.update(loss=losses.item(), **loss_values)
            metric_logger.update(lr=optimizer.param_groups[0]['lr'])
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}: loss={losses.item():.4f}")
            
            batch_idx += 1
        
        # 更新学习率
        scheduler.step()
        
        # 打印训练统计
        avg_loss = metric_logger.meters['loss'].global_avg
        print(f"  Train Loss: {avg_loss:.4f}")
        logger.info(f"Epoch {epoch + 1}: train_loss={avg_loss:.4f}")
        
        # 验证
        if (epoch + 1) % args.eval_interval == 0:
            print("\n  验证中...")
            model.eval()
            text_encoder.eval()
            
            val_loss = 0
            val_batches = 0
            
            with jt.no_grad():
                for samples, targets in val_loader:
                    texts = [t.get('text', '') for t in targets]
                    text_dict = text_encoder(texts)
                    outputs = model(samples, text_dict)
                    loss_dict = criterion(outputs, targets)
                    losses = sum(loss_dict[k] * criterion.weight_dict.get(k, 1.0) 
                                for k in loss_dict.keys())
                    val_loss += losses.item()
                    val_batches += 1
            
            val_loss /= max(val_batches, 1)
            print(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"Epoch {epoch + 1}: val_loss={val_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_model(
                model, optimizer, scheduler,
                epoch, avg_loss,
                checkpoint_dir, "groundingdino"
            )
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(
                model, optimizer, scheduler,
                epoch, avg_loss,
                checkpoint_dir, "groundingdino_best"
            )
    
    # 保存最终模型
    save_model(
        model, optimizer, scheduler,
        args.epochs - 1, avg_loss,
        checkpoint_dir, "groundingdino_final"
    )
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"检查点保存在: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
