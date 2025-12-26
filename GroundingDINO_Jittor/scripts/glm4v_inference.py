#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GLM-4.1V-9B-Thinking 双卡 3090 图像识别脚本

环境配置:
    pip install transformers>=4.57.1 torch accelerate pillow

用法:
    python glm4v_inference.py --image test.jpg --prompt "描述这张图片"
    python glm4v_inference.py --image test.jpg --prompt "这张图片中有什么物体？"
"""

import os
import argparse
import torch
import json
import time
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, Glm4vForConditionalGeneration

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "zai-org/GLM-4.1V-9B-Thinking"

class Config:
    def __init__(self):
        self.max_new_tokens = 4096
        self.temperature = 0.7
        self.do_sample = True
        self.dtype = torch.bfloat16  # 2x3090 用 bf16 节省显存

# ============================================================
# 模型加载 (双卡自动分配)
# ============================================================
def load_model(config: Config):
    """加载模型，自动分配到双卡"""
    print(f"Loading model from {MODEL_PATH}...")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    
    # device_map="auto" 自动在多卡间分配模型层
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=config.dtype,
        device_map="auto",  # 关键：自动多卡分配
        trust_remote_code=True,
    )
    
    model.eval()
    print("Model loaded successfully")
    print(f"Model device map: {model.hf_device_map}")
    
    return model, processor

# ============================================================
# 推理
# ============================================================
def inference(
    model,
    processor,
    image_path: str,
    prompt: str,
    config: Config
) -> str:
    """单张图像推理"""
    
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # 处理输入
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.do_sample,
        )
    
    # 解码输出 (去掉输入部分)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
    return response

# ============================================================
# 批量推理
# ============================================================
def batch_inference(
    model,
    processor,
    image_paths: list,
    prompt: str,
    config: Config,
    output_dir: str = None
) -> list:
    """批量图像推理"""
    results = []
    start_time = time.time()

    for i, img_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")
        image_start_time = time.time()

        try:
            # GPU memory monitoring
            if torch.cuda.is_available():
                gpu_memory_before = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3

            response = inference(model, processor, img_path, prompt, config)

            image_time = time.time() - image_start_time

            result = {
                "image_path": img_path,
                "image_filename": os.path.basename(img_path),
                "prompt": prompt,
                "response": response,
                "inference_time_seconds": round(image_time, 2),
                "timestamp": datetime.now().isoformat(),
                "gpu_memory_gb": round(gpu_memory_before, 2) if torch.cuda.is_available() else 0.0,
                "success": True
            }

            results.append(result)
            print(f"✓ Completed in {image_time:.2f}s - Response length: {len(response)} chars")

            # Save individual result if output_dir specified
            if output_dir:
                result_file = os.path.join(output_dir, f"result_{i+1:03d}_{os.path.basename(img_path)}.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

        except Exception as e:
            error_time = time.time() - image_start_time
            print(f"✗ Error after {error_time:.2f}s: {e}")

            result = {
                "image_path": img_path,
                "image_filename": os.path.basename(img_path),
                "prompt": prompt,
                "error": str(e),
                "inference_time_seconds": round(error_time, 2),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            results.append(result)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Batch processing completed in {total_time:.2f} seconds")
    print(f"Successful: {len([r for r in results if r['success']])}/{len(results)}")
    print(f"Average time per image: {total_time/len(results):.2f}s")

    return results

# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="GLM-4.1V-9B-Thinking Inference")
    parser.add_argument("--image", type=str, required=True,
                        help="Image path or directory")
    parser.add_argument("--prompt", type=str, default="请详细描述这张图片的内容。",
                        help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for JSON results")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Limit number of images to process (for testing)")
    args = parser.parse_args()

    # 配置
    config = Config()
    config.max_new_tokens = args.max_tokens
    config.temperature = args.temperature

    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Results will be saved to: {args.output_dir}")

    # 加载模型
    model, processor = load_model(config)

    # 推理
    if os.path.isdir(args.image):
        # 批量处理目录
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        all_images = [os.path.join(args.image, f) for f in sorted(os.listdir(args.image))
                      if os.path.splitext(f)[1].lower() in exts]

        # 限制图像数量
        if args.num_images:
            all_images = all_images[:args.num_images]
            print(f"Processing first {args.num_images} images out of {len(all_images) + (len([f for f in os.listdir(args.image) if os.path.splitext(f)[1].lower() in exts]) - len(all_images))} total")

        print(f"Found {len(all_images)} images to process")

        results = batch_inference(model, processor, all_images, args.prompt, config, args.output_dir)

        # 保存汇总结果
        if args.output_dir:
            summary_file = os.path.join(args.output_dir, "summary.json")
            summary = {
                "total_images": len(results),
                "successful": len([r for r in results if r["success"]]),
                "failed": len([r for r in results if not r["success"]]),
                "total_time_seconds": sum(r.get("inference_time_seconds", 0) for r in results),
                "average_time_per_image": sum(r.get("inference_time_seconds", 0) for r in results) / len(results),
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "model": MODEL_PATH,
                    "prompt": args.prompt,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "dtype": str(config.dtype)
                },
                "results": results
            }

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"\nSummary saved to: {summary_file}")

        # 显示结果摘要
        for r in results:
            print(f"\n{'='*60}")
            print(f"Image: {r['image_filename']}")
            if r['success']:
                print(f"Time: {r['inference_time_seconds']}s")
                print(f"Response preview: {r['response'][:200]}...")
            else:
                print(f"Error: {r.get('error', 'Unknown error')}")

    else:
        # 单张图像
        response = inference(model, processor, args.image, args.prompt, config)
        print(f"\n{'='*60}")
        print(f"Image: {args.image}")
        print(f"Prompt: {args.prompt}")
        print(f"{'='*60}")
        print(response)

if __name__ == "__main__":
    main()
