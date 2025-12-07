#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grounding DINO Jittor - 完整推理示例

用法:
    python scripts/run_inference.py --image <图像路径> --text "cat . dog ." --output output.jpg
"""

import os
import sys
import argparse
from PIL import Image
import numpy as np

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import jittor as jt
from jittor import nn


def create_test_image(output_path: str = "test_image.jpg", size: tuple = (640, 480)):
    """创建测试图像"""
    # 创建一个简单的测试图像
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # 添加一些彩色方块模拟物体
    # 红色方块 (模拟 "object 1")
    img[100:200, 100:250, 0] = 200
    img[100:200, 100:250, 1] = 50
    img[100:200, 100:250, 2] = 50
    
    # 绿色方块 (模拟 "object 2")
    img[250:400, 300:500, 0] = 50
    img[250:400, 300:500, 1] = 200
    img[250:400, 300:500, 2] = 50
    
    # 蓝色方块 (模拟 "object 3")
    img[150:280, 450:580, 0] = 50
    img[150:280, 450:580, 1] = 50
    img[150:280, 450:580, 2] = 200
    
    # 背景渐变
    for y in range(size[1]):
        for x in range(size[0]):
            if img[y, x].sum() == 0:  # 只修改背景
                img[y, x] = [220 + int(20 * y / size[1]), 
                            220 + int(20 * x / size[0]), 
                            230]
    
    # 保存
    Image.fromarray(img).save(output_path)
    print(f"Created test image: {output_path}")
    return output_path


def run_inference_demo():
    """运行推理演示"""
    print("=" * 60)
    print("Grounding DINO Jittor - 完整推理演示")
    print("=" * 60)
    
    # 设置路径
    weight_path = os.path.join(project_root, "weights", "groundingdino_swint_ogc_jittor.pkl")
    
    # 检查权重文件
    if not os.path.exists(weight_path):
        print(f"\n错误: 权重文件不存在: {weight_path}")
        print("\n请先转换权重:")
        print("  python scripts/convert_weights_pytorch_to_jittor.py \\")
        print("      --pytorch_weight weights/groundingdino_swint_ogc.pth \\")
        print("      --output weights/groundingdino_swint_ogc_jittor.pkl")
        return False
    
    # 创建测试图像
    test_image_path = os.path.join(project_root, "test_image.jpg")
    if not os.path.exists(test_image_path):
        create_test_image(test_image_path)
    
    # 测试文本
    test_caption = "red box . green box . blue box ."
    
    print(f"\n测试配置:")
    print(f"  权重: {weight_path}")
    print(f"  图像: {test_image_path}")
    print(f"  文本: {test_caption}")
    
    # ============================================================
    # 方法1: 使用推理工具类
    # ============================================================
    print("\n" + "=" * 60)
    print("[方法1] 使用 GroundingDINOInference 类")
    print("=" * 60)
    
    try:
        from jittor_implementation.util.inference import (
            GroundingDINOInference,
            load_image,
            plot_boxes_to_image,
        )
        
        # 初始化模型
        print("\n正在初始化模型...")
        model = GroundingDINOInference(
            weight_path=weight_path,
            device="cuda",
            box_threshold=0.3,
            text_threshold=0.2,
        )
        
        # 执行推理
        print("\n正在推理...")
        boxes, scores, phrases = model.predict(
            image=test_image_path,
            caption=test_caption,
        )
        
        print(f"\n检测结果:")
        print(f"  检测到 {len(boxes)} 个目标")
        for i, (box, score, phrase) in enumerate(zip(boxes, scores, phrases)):
            print(f"  {i+1}. {phrase}: {score:.3f}")
            print(f"      边界框: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]")
        
        # 可视化
        output_path = os.path.join(project_root, "output_inference.jpg")
        result_image = model.predict_and_visualize(
            image_path=test_image_path,
            caption=test_caption,
            output_path=output_path,
        )
        
        print(f"\n✓ 推理完成!")
        print(f"  输出图像: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_with_custom_image(image_path: str, caption: str, output_path: str = "output.jpg"):
    """使用自定义图像运行推理"""
    print("=" * 60)
    print("Grounding DINO Jittor - 自定义图像推理")
    print("=" * 60)
    
    weight_path = os.path.join(project_root, "weights", "groundingdino_swint_ogc_jittor.pkl")
    
    if not os.path.exists(weight_path):
        print(f"错误: 权重文件不存在: {weight_path}")
        return False
    
    if not os.path.exists(image_path):
        print(f"错误: 图像不存在: {image_path}")
        return False
    
    try:
        from jittor_implementation.util.inference import GroundingDINOInference
        
        print(f"\n输入图像: {image_path}")
        print(f"文本提示: {caption}")
        
        # 初始化模型
        model = GroundingDINOInference(
            weight_path=weight_path,
            device="cuda",
        )
        
        # 推理并可视化
        result = model.predict_and_visualize(
            image_path=image_path,
            caption=caption,
            output_path=output_path,
        )
        
        print(f"\n✓ 完成! 输出: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Grounding DINO Jittor 推理')
    parser.add_argument('--image', '-i', type=str, default=None, help='输入图像路径')
    parser.add_argument('--text', '-t', type=str, default="object .", help='文本提示')
    parser.add_argument('--output', '-o', type=str, default='output.jpg', help='输出路径')
    parser.add_argument('--demo', action='store_true', help='运行演示')
    parser.add_argument('--box_threshold', type=float, default=0.35, help='边界框阈值')
    parser.add_argument('--text_threshold', type=float, default=0.25, help='文本阈值')
    
    args = parser.parse_args()
    
    if args.demo or args.image is None:
        # 运行演示
        run_inference_demo()
    else:
        # 使用自定义图像
        run_with_custom_image(args.image, args.text, args.output)


if __name__ == '__main__':
    main()

