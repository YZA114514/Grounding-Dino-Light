# ------------------------------------------------------------------------
# Grounding DINO - Jittor Implementation
# 下载并转换官方预训练权重
# ------------------------------------------------------------------------
"""
下载并转换 Grounding DINO 官方预训练权重

使用方法:
    python download_and_convert_weights.py

这个脚本会：
1. 创建 weights 目录
2. 下载官方预训练权重 groundingdino_swint_ogc.pth
3. 分析权重结构
4. 转换为 Jittor 格式
5. 保存到 weights/groundingdino_swint_ogc_jittor.pkl
"""

import os
import sys
import urllib.request
import pickle
from collections import OrderedDict

import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 配置
# ============================================================

# 官方预训练权重下载链接
WEIGHT_URLS = {
    "groundingdino_swint_ogc": {
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "description": "Grounding DINO with Swin-T backbone (recommended)",
        "size_mb": 694,
    },
    "groundingdino_swinb_cogcoor": {
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
        "description": "Grounding DINO with Swin-B backbone (larger)",
        "size_mb": 938,
    },
}

# 默认下载的模型
DEFAULT_MODEL = "groundingdino_swint_ogc"

# 权重目录
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


# ============================================================
# 下载函数
# ============================================================

def download_with_progress(url: str, save_path: str) -> bool:
    """
    带进度条的下载函数
    """
    print(f"正在下载: {url}")
    print(f"保存到: {save_path}")
    
    try:
        # 获取文件大小
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get('content-length', 0))
        
        # 下载
        downloaded = 0
        block_size = 8192
        
        with open(save_path, 'wb') as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                f.write(buffer)
                
                # 显示进度
                if total_size > 0:
                    progress = downloaded / total_size * 100
                    bar_length = 50
                    filled = int(bar_length * downloaded / total_size)
                    bar = '=' * filled + '-' * (bar_length - filled)
                    print(f"\r[{bar}] {progress:.1f}% ({downloaded/1e6:.1f}/{total_size/1e6:.1f} MB)", end='')
        
        print("\n下载完成！")
        return True
        
    except Exception as e:
        print(f"\n下载失败: {e}")
        return False


def download_weights(model_name: str = DEFAULT_MODEL, force: bool = False) -> str:
    """
    下载预训练权重
    
    Args:
        model_name: 模型名称
        force: 是否强制重新下载
        
    Returns:
        下载的文件路径
    """
    if model_name not in WEIGHT_URLS:
        available = ", ".join(WEIGHT_URLS.keys())
        raise ValueError(f"未知模型: {model_name}，可用模型: {available}")
    
    info = WEIGHT_URLS[model_name]
    
    # 创建权重目录
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    # 下载路径
    filename = f"{model_name}.pth"
    save_path = os.path.join(WEIGHTS_DIR, filename)
    
    # 检查是否已存在
    if os.path.exists(save_path) and not force:
        print(f"权重文件已存在: {save_path}")
        print("使用 --force 强制重新下载")
        return save_path
    
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"描述: {info['description']}")
    print(f"大小: ~{info['size_mb']} MB")
    print(f"{'='*60}\n")
    
    # 下载
    success = download_with_progress(info['url'], save_path)
    
    if not success:
        raise RuntimeError("下载失败")
    
    return save_path


# ============================================================
# 权重分析函数
# ============================================================

def analyze_weights(weight_path: str) -> dict:
    """
    分析 PyTorch 权重结构
    
    Returns:
        权重结构信息字典
    """
    try:
        import torch
    except ImportError:
        print("需要安装 PyTorch 来分析权重")
        return {}
    
    print(f"\n分析权重文件: {weight_path}")
    
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    # 获取 state_dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("Checkpoint 格式: {'model': state_dict, ...}")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Checkpoint 格式: {'state_dict': state_dict, ...}")
        else:
            state_dict = checkpoint
            print("Checkpoint 格式: state_dict")
        
        # 打印其他键
        other_keys = [k for k in checkpoint.keys() if k not in ['model', 'state_dict']]
        if other_keys:
            print(f"其他键: {other_keys}")
    else:
        state_dict = checkpoint
    
    # 分析权重结构
    analysis = {
        "total_params": 0,
        "modules": {},
        "weight_names": [],
    }
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            size = param.numel()
            analysis["total_params"] += size
            analysis["weight_names"].append(name)
            
            # 解析模块名
            parts = name.split('.')
            module = parts[0]
            if module not in analysis["modules"]:
                analysis["modules"][module] = {"count": 0, "params": 0, "weights": []}
            analysis["modules"][module]["count"] += 1
            analysis["modules"][module]["params"] += size
            analysis["modules"][module]["weights"].append(name)
    
    # 打印分析结果
    print(f"\n{'='*60}")
    print("权重结构分析")
    print(f"{'='*60}")
    print(f"总参数量: {analysis['total_params']:,} ({analysis['total_params']/1e6:.2f}M)")
    print(f"权重数量: {len(analysis['weight_names'])}")
    
    print(f"\n模块统计:")
    print(f"{'模块名':<30} {'权重数':<10} {'参数量':<15}")
    print("-" * 60)
    for module, info in sorted(analysis["modules"].items(), key=lambda x: -x[1]["params"]):
        params_str = f"{info['params']:,} ({info['params']/1e6:.2f}M)"
        print(f"{module:<30} {info['count']:<10} {params_str:<15}")
    
    return analysis


def print_weight_structure(weight_path: str, max_items: int = 50):
    """
    打印权重的详细结构
    """
    try:
        import torch
    except ImportError:
        print("需要安装 PyTorch")
        return
    
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"\n权重列表 (前 {max_items} 个):")
    print(f"{'名称':<60} {'形状':<20} {'参数量':<15}")
    print("-" * 95)
    
    for i, (name, param) in enumerate(state_dict.items()):
        if i >= max_items:
            print(f"... 还有 {len(state_dict) - max_items} 个")
            break
        
        if isinstance(param, torch.Tensor):
            shape = str(tuple(param.shape))
            params = param.numel()
            print(f"{name:<60} {shape:<20} {params:,}")


# ============================================================
# 转换函数
# ============================================================

def convert_pytorch_to_jittor(pytorch_path: str, jittor_path: str = None) -> str:
    """
    将 PyTorch 权重转换为 Jittor 格式
    
    Args:
        pytorch_path: PyTorch 权重路径
        jittor_path: Jittor 权重保存路径（默认为同目录下的 _jittor.pkl）
        
    Returns:
        Jittor 权重路径
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("需要安装 PyTorch 来转换权重")
    
    if jittor_path is None:
        base = os.path.splitext(pytorch_path)[0]
        jittor_path = f"{base}_jittor.pkl"
    
    print(f"\n{'='*60}")
    print("转换 PyTorch → Jittor")
    print(f"{'='*60}")
    print(f"输入: {pytorch_path}")
    print(f"输出: {jittor_path}")
    
    # 加载 PyTorch 权重
    print("\n加载 PyTorch 权重...")
    checkpoint = torch.load(pytorch_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 转换为 numpy
    print("转换为 NumPy 格式...")
    numpy_weights = OrderedDict()
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            numpy_weights[name] = param.detach().cpu().numpy()
        else:
            numpy_weights[name] = np.array(param)
    
    # 保存
    print("保存 Jittor 权重...")
    with open(jittor_path, 'wb') as f:
        pickle.dump(numpy_weights, f)
    
    # 验证
    print("验证转换...")
    with open(jittor_path, 'rb') as f:
        loaded = pickle.load(f)
    
    assert len(loaded) == len(numpy_weights), "权重数量不匹配"
    
    for name in numpy_weights:
        assert name in loaded, f"缺少权重: {name}"
        assert np.allclose(numpy_weights[name], loaded[name]), f"权重值不匹配: {name}"
    
    print(f"\n✓ 转换成功！")
    print(f"  权重数量: {len(numpy_weights)}")
    print(f"  保存到: {jittor_path}")
    
    return jittor_path


# ============================================================
# 权重加载辅助函数（供 Jittor 模型使用）
# ============================================================

def load_pretrained_weights(model, weight_path: str, freeze: bool = True, strict: bool = False):
    """
    加载预训练权重到 Jittor 模型
    
    Args:
        model: Jittor 模型
        weight_path: 权重文件路径 (.pkl 或 .pth)
        freeze: 是否冻结预训练权重
        strict: 是否严格匹配
        
    Returns:
        (matched, missing, unexpected) 匹配/缺失/多余的权重名称
    """
    try:
        import jittor as jt
    except ImportError:
        raise RuntimeError("需要安装 Jittor")
    
    print(f"\n加载预训练权重: {weight_path}")
    
    # 加载权重
    if weight_path.endswith('.pkl'):
        with open(weight_path, 'rb') as f:
            weights = pickle.load(f)
    else:
        # PyTorch 格式，先转换
        import torch
        checkpoint = torch.load(weight_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        weights = {k: v.numpy() for k, v in state_dict.items()}
    
    # 获取模型参数
    model_state = model.state_dict()
    
    matched = []
    missing = []
    unexpected = []
    
    # 加载匹配的权重
    for name, param in model_state.items():
        if name in weights:
            weight = weights[name]
            if param.shape == tuple(weight.shape):
                param.update(jt.array(weight))
                matched.append(name)
                
                # 冻结权重
                if freeze:
                    param.stop_grad()
            else:
                print(f"  形状不匹配: {name} 模型{param.shape} vs 权重{weight.shape}")
                missing.append(name)
        else:
            missing.append(name)
    
    # 检查多余的权重
    for name in weights:
        if name not in model_state:
            unexpected.append(name)
    
    print(f"\n加载结果:")
    print(f"  ✓ 匹配: {len(matched)}")
    print(f"  ✗ 缺失: {len(missing)}")
    print(f"  ? 多余: {len(unexpected)}")
    
    if freeze:
        print(f"  ❄ 已冻结 {len(matched)} 个预训练权重")
    
    if strict and (missing or unexpected):
        raise RuntimeError("严格模式下权重不完全匹配")
    
    return matched, missing, unexpected


def freeze_module(module, freeze: bool = True):
    """
    冻结/解冻模块的所有参数
    
    Args:
        module: Jittor 模块
        freeze: True=冻结, False=解冻
    """
    try:
        import jittor as jt
    except ImportError:
        raise RuntimeError("需要安装 Jittor")
    
    for param in module.parameters():
        if freeze:
            param.stop_grad()
        else:
            param.start_grad()
    
    status = "冻结" if freeze else "解冻"
    param_count = sum(1 for _ in module.parameters())
    print(f"已{status} {param_count} 个参数")


def get_trainable_params(model):
    """
    获取可训练参数
    
    Returns:
        可训练参数列表
    """
    trainable = []
    frozen = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append((name, param))
        else:
            frozen.append((name, param))
    
    print(f"可训练参数: {len(trainable)}")
    print(f"冻结参数: {len(frozen)}")
    
    return trainable


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="下载并转换 Grounding DINO 预训练权重")
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(WEIGHT_URLS.keys()),
        help=f"模型名称，默认: {DEFAULT_MODEL}"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制重新下载"
    )
    
    parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="分析权重结构"
    )
    
    parser.add_argument(
        "--convert", "-c",
        action="store_true",
        help="转换为 Jittor 格式"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="执行所有操作（下载+分析+转换）"
    )
    
    args = parser.parse_args()
    
    if args.all:
        args.analyze = True
        args.convert = True
    
    # 如果没有指定任何操作，默认执行全部
    if not (args.analyze or args.convert):
        args.analyze = True
        args.convert = True
    
    # 下载
    pytorch_path = download_weights(args.model, args.force)
    
    # 分析
    if args.analyze:
        analyze_weights(pytorch_path)
        print_weight_structure(pytorch_path, max_items=30)
    
    # 转换
    if args.convert:
        jittor_path = convert_pytorch_to_jittor(pytorch_path)
        print(f"\n转换完成！Jittor 权重保存到: {jittor_path}")


if __name__ == "__main__":
    main()



