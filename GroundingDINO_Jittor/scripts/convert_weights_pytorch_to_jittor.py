# ------------------------------------------------------------------------
# Grounding DINO - Jittor Implementation
# Weight Conversion Script: PyTorch → Jittor (Member A)
# ------------------------------------------------------------------------
"""
权重转换脚本说明：

本脚本用于将 PyTorch 预训练权重转换为 Jittor 格式。

主要功能：
1. 加载 PyTorch 权重文件 (.pth, .pt, .bin)
2. 转换张量格式 (torch.Tensor → numpy → jittor.Var)
3. 处理权重名称映射（如果需要）
4. 保存为 Jittor 格式 (.pkl)
5. 验证转换正确性

使用方法：
    python convert_weights_pytorch_to_jittor.py \
        --pytorch_weight path/to/pytorch_model.pth \
        --output_path path/to/jittor_model.pkl \
        --verify

官方预训练权重下载：
    https://github.com/IDEA-Research/GroundingDINO
    - groundingdino_swint_ogc.pth (Swin-T backbone)
    - groundingdino_swinb_cogcoor.pth (Swin-B backbone)
"""

import os
import sys
import argparse
import pickle
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np

# 尝试导入 PyTorch（用于加载权重）
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: PyTorch 未安装，部分功能可能不可用")

# 尝试导入 Jittor（用于验证）
try:
    import jittor as jt
    HAS_JITTOR = True
except ImportError:
    HAS_JITTOR = False
    print("警告: Jittor 未安装，无法进行验证")


# ============================================================
# 权重名称映射表
# ============================================================

# PyTorch 和 Jittor 模块名称映射
# 格式: "pytorch_name_pattern": "jittor_name_pattern"
NAME_MAPPING = {
    # 通常不需要映射，因为我们的 Jittor 实现保持了相同的命名
    # 如果有不同的命名，在这里添加映射规则
    
    # 示例：
    # "module.": "",  # 去除 DataParallel 的 module. 前缀
    # "backbone.body.": "backbone.",  # 简化 backbone 路径
}

# 需要跳过的权重（如果有）
SKIP_WEIGHTS = [
    # 例如：某些只在训练时使用的权重
    # "some_training_only_weight",
]

# 需要转置的权重（某些情况下 PyTorch 和其他框架的权重布局不同）
TRANSPOSE_WEIGHTS = [
    # 例如：某些全连接层权重
    # "some_fc_layer.weight",
]


# ============================================================
# 核心转换函数
# ============================================================

def load_pytorch_weights(weight_path: str) -> Dict[str, np.ndarray]:
    """
    加载 PyTorch 权重文件
    
    Args:
        weight_path: PyTorch 权重文件路径 (.pth, .pt, .bin)
        
    Returns:
        权重字典 {name: numpy_array}
    """
    if not HAS_TORCH:
        raise RuntimeError("需要安装 PyTorch 才能加载 .pth 文件")
    
    print(f"正在加载 PyTorch 权重: {weight_path}")
    
    # 加载权重
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    # 处理不同的 checkpoint 格式
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'module' in checkpoint:
            state_dict = checkpoint['module']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 转换为 numpy
    numpy_weights = OrderedDict()
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            # 去除 module. 前缀（来自 DDP 训练）
            clean_name = name.replace("module.", "") if name.startswith("module.") else name
            numpy_weights[clean_name] = tensor.detach().cpu().numpy()
        else:
            # 跳过非张量对象（如配置对象）
            print(f"跳过非张量对象: {name} (类型: {type(tensor).__name__})")
    
    print(f"成功加载 {len(numpy_weights)} 个权重")
    
    return numpy_weights


def convert_weight_names(
    weights: Dict[str, np.ndarray],
    name_mapping: Optional[Dict[str, str]] = None,
    skip_weights: Optional[list] = None,
) -> Dict[str, np.ndarray]:
    """
    转换权重名称
    
    Args:
        weights: 原始权重字典
        name_mapping: 名称映射规则
        skip_weights: 需要跳过的权重名称
        
    Returns:
        转换后的权重字典
    """
    if name_mapping is None:
        name_mapping = NAME_MAPPING
    if skip_weights is None:
        skip_weights = SKIP_WEIGHTS
    
    converted = OrderedDict()
    skipped = []
    
    for name, weight in weights.items():
        # 检查是否需要跳过
        should_skip = False
        for skip_pattern in skip_weights:
            if skip_pattern in name:
                should_skip = True
                skipped.append(name)
                break
        
        if should_skip:
            continue
        
        # 应用名称映射
        new_name = name
        for old_pattern, new_pattern in name_mapping.items():
            new_name = new_name.replace(old_pattern, new_pattern)
        
        converted[new_name] = weight
    
    if skipped:
        print(f"跳过了 {len(skipped)} 个权重:")
        for name in skipped[:5]:  # 只显示前5个
            print(f"  - {name}")
        if len(skipped) > 5:
            print(f"  ... 还有 {len(skipped) - 5} 个")
    
    return converted


def transpose_weights(
    weights: Dict[str, np.ndarray],
    transpose_list: Optional[list] = None,
) -> Dict[str, np.ndarray]:
    """
    转置特定的权重
    
    某些情况下，不同框架的权重布局不同，需要转置
    
    Args:
        weights: 权重字典
        transpose_list: 需要转置的权重名称列表
        
    Returns:
        处理后的权重字典
    """
    if transpose_list is None:
        transpose_list = TRANSPOSE_WEIGHTS
    
    for name in transpose_list:
        if name in weights:
            print(f"转置权重: {name}")
            weights[name] = weights[name].T
    
    return weights


def save_jittor_weights(
    weights: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """
    保存为 Jittor 格式
    
    Args:
        weights: 权重字典 (numpy 格式)
        output_path: 输出路径
    """
    print(f"正在保存 Jittor 权重: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Jittor 使用 pickle 格式保存权重
    with open(output_path, 'wb') as f:
        pickle.dump(weights, f)
    
    print(f"成功保存 {len(weights)} 个权重")


def load_jittor_weights(weight_path: str) -> Dict[str, np.ndarray]:
    """
    加载 Jittor 权重文件
    
    Args:
        weight_path: Jittor 权重文件路径 (.pkl)
        
    Returns:
        权重字典
    """
    print(f"正在加载 Jittor 权重: {weight_path}")
    
    with open(weight_path, 'rb') as f:
        weights = pickle.load(f)
    
    print(f"成功加载 {len(weights)} 个权重")
    
    return weights


# ============================================================
# 验证函数
# ============================================================

def verify_conversion(
    pytorch_weights: Dict[str, np.ndarray],
    jittor_weights: Dict[str, np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[bool, list]:
    """
    验证转换是否正确
    
    Args:
        pytorch_weights: 原始 PyTorch 权重
        jittor_weights: 转换后的 Jittor 权重
        rtol: 相对误差容限
        atol: 绝对误差容限
        
    Returns:
        (是否通过, 不匹配的权重列表)
    """
    print("\n正在验证转换...")
    
    mismatched = []
    
    for name, pt_weight in pytorch_weights.items():
        if name not in jittor_weights:
            # 可能是名称映射后的名称不同
            mismatched.append((name, "未找到对应权重"))
            continue
        
        jt_weight = jittor_weights[name]
        
        # 检查形状
        if pt_weight.shape != jt_weight.shape:
            mismatched.append((name, f"形状不匹配: {pt_weight.shape} vs {jt_weight.shape}"))
            continue
        
        # 检查数值
        if not np.allclose(pt_weight, jt_weight, rtol=rtol, atol=atol):
            max_diff = np.abs(pt_weight - jt_weight).max()
            mismatched.append((name, f"数值不匹配, 最大差异: {max_diff}"))
    
    passed = len(mismatched) == 0
    
    if passed:
        print("✓ 验证通过！所有权重转换正确")
    else:
        print(f"✗ 验证失败！{len(mismatched)} 个权重不匹配:")
        for name, reason in mismatched[:10]:
            print(f"  - {name}: {reason}")
        if len(mismatched) > 10:
            print(f"  ... 还有 {len(mismatched) - 10} 个")
    
    return passed, mismatched


def print_weight_info(weights: Dict[str, np.ndarray], title: str = "权重信息"):
    """
    打印权重信息
    """
    print(f"\n{title}")
    print("=" * 60)
    
    total_params = 0
    for name, weight in weights.items():
        params = np.prod(weight.shape)
        total_params += params
    
    print(f"权重数量: {len(weights)}")
    print(f"总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    print("\n前10个权重:")
    for i, (name, weight) in enumerate(weights.items()):
        if i >= 10:
            break
        print(f"  {name}: {weight.shape} ({weight.dtype})")
    
    if len(weights) > 10:
        print(f"  ... 还有 {len(weights) - 10} 个")


# ============================================================
# 高级功能
# ============================================================

def convert_bert_weights(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    专门处理 BERT 权重的转换
    
    Hugging Face BERT 和其他框架的 BERT 可能有不同的命名
    """
    converted = OrderedDict()
    
    bert_mapping = {
        # Hugging Face BERT → 通用命名
        "bert.embeddings.word_embeddings.weight": "word_embeddings.weight",
        "bert.embeddings.position_embeddings.weight": "position_embeddings.weight",
        "bert.embeddings.token_type_embeddings.weight": "token_type_embeddings.weight",
        "bert.embeddings.LayerNorm.weight": "embeddings_LayerNorm.weight",
        "bert.embeddings.LayerNorm.bias": "embeddings_LayerNorm.bias",
    }
    
    for name, weight in weights.items():
        new_name = bert_mapping.get(name, name)
        converted[new_name] = weight
    
    return converted


def extract_backbone_weights(
    weights: Dict[str, np.ndarray],
    prefix: str = "backbone.",
) -> Dict[str, np.ndarray]:
    """
    提取 backbone 权重
    
    有时候只需要 backbone 的权重（如用于迁移学习）
    """
    backbone_weights = OrderedDict()
    
    for name, weight in weights.items():
        if name.startswith(prefix):
            # 去除前缀
            new_name = name[len(prefix):]
            backbone_weights[new_name] = weight
    
    print(f"提取了 {len(backbone_weights)} 个 backbone 权重")
    
    return backbone_weights


def merge_weights(
    base_weights: Dict[str, np.ndarray],
    new_weights: Dict[str, np.ndarray],
    overwrite: bool = True,
) -> Dict[str, np.ndarray]:
    """
    合并权重
    
    用于将预训练权重和微调权重合并
    """
    merged = OrderedDict(base_weights)
    
    added = 0
    updated = 0
    
    for name, weight in new_weights.items():
        if name in merged:
            if overwrite:
                merged[name] = weight
                updated += 1
        else:
            merged[name] = weight
            added += 1
    
    print(f"合并完成: 新增 {added} 个, 更新 {updated} 个")
    
    return merged


# ============================================================
# Jittor 模型加载辅助函数
# ============================================================

def load_weights_to_jittor_model(model, weight_path: str, strict: bool = True):
    """
    将权重加载到 Jittor 模型
    
    Args:
        model: Jittor 模型实例
        weight_path: 权重文件路径
        strict: 是否严格匹配（所有权重必须匹配）
        
    Returns:
        (匹配的权重, 未匹配的权重, 多余的权重)
    """
    if not HAS_JITTOR:
        raise RuntimeError("需要安装 Jittor")
    
    # 加载权重
    if weight_path.endswith('.pkl'):
        weights = load_jittor_weights(weight_path)
    else:
        # 假设是 PyTorch 格式，先转换
        weights = load_pytorch_weights(weight_path)
    
    # 获取模型的状态字典
    model_state = model.state_dict()
    
    matched = []
    missing = []
    unexpected = []
    
    # 加载权重
    for name, param in model_state.items():
        if name in weights:
            weight = weights[name]
            if param.shape == weight.shape:
                param.update(jt.array(weight))
                matched.append(name)
            else:
                print(f"警告: {name} 形状不匹配: 模型 {param.shape} vs 权重 {weight.shape}")
                missing.append(name)
        else:
            missing.append(name)
    
    # 检查多余的权重
    for name in weights:
        if name not in model_state:
            unexpected.append(name)
    
    print(f"\n权重加载结果:")
    print(f"  匹配: {len(matched)}")
    print(f"  缺失: {len(missing)}")
    print(f"  多余: {len(unexpected)}")
    
    if strict and (missing or unexpected):
        raise RuntimeError("严格模式下权重不完全匹配")
    
    return matched, missing, unexpected


# ============================================================
# 主函数
# ============================================================

def convert(
    pytorch_weight_path: str,
    output_path: str,
    verify: bool = True,
    print_info: bool = True,
) -> bool:
    """
    执行完整的转换流程
    
    Args:
        pytorch_weight_path: PyTorch 权重路径
        output_path: 输出路径
        verify: 是否验证转换
        print_info: 是否打印权重信息
        
    Returns:
        是否成功
    """
    print("=" * 60)
    print("PyTorch → Jittor 权重转换工具")
    print("=" * 60)
    
    # 1. 加载 PyTorch 权重
    pytorch_weights = load_pytorch_weights(pytorch_weight_path)
    
    if print_info:
        print_weight_info(pytorch_weights, "PyTorch 权重信息")
    
    # 2. 转换权重名称
    converted_weights = convert_weight_names(pytorch_weights)
    
    # 3. 处理需要转置的权重
    converted_weights = transpose_weights(converted_weights)
    
    # 4. 保存 Jittor 权重
    save_jittor_weights(converted_weights, output_path)
    
    # 5. 验证转换
    if verify:
        loaded_weights = load_jittor_weights(output_path)
        passed, _ = verify_conversion(pytorch_weights, loaded_weights)
        if not passed:
            return False
    
    print("\n" + "=" * 60)
    print("转换完成！")
    print(f"输出文件: {output_path}")
    print("=" * 60)
    
    return True


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="PyTorch → Jittor 权重转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本转换
    python convert_weights_pytorch_to_jittor.py \\
        --pytorch_weight groundingdino_swint_ogc.pth \\
        --output jittor_groundingdino_swint.pkl
    
    # 转换并验证
    python convert_weights_pytorch_to_jittor.py \\
        --pytorch_weight groundingdino_swint_ogc.pth \\
        --output jittor_groundingdino_swint.pkl \\
        --verify
    
    # 只打印权重信息
    python convert_weights_pytorch_to_jittor.py \\
        --pytorch_weight groundingdino_swint_ogc.pth \\
        --info_only
        """
    )
    
    parser.add_argument(
        "--pytorch_weight", "-p",
        type=str,
        required=True,
        help="PyTorch 权重文件路径 (.pth, .pt, .bin)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径 (.pkl)，默认为输入文件名 + _jittor.pkl"
    )
    
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="验证转换是否正确"
    )
    
    parser.add_argument(
        "--info_only", "-i",
        action="store_true",
        help="只打印权重信息，不转换"
    )
    
    parser.add_argument(
        "--extract_backbone",
        action="store_true",
        help="只提取 backbone 权重"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.pytorch_weight):
        print(f"错误: 文件不存在: {args.pytorch_weight}")
        sys.exit(1)
    
    # 只打印信息
    if args.info_only:
        weights = load_pytorch_weights(args.pytorch_weight)
        print_weight_info(weights)
        sys.exit(0)
    
    # 确定输出路径
    if args.output is None:
        base_name = os.path.splitext(args.pytorch_weight)[0]
        args.output = f"{base_name}_jittor.pkl"
    
    # 执行转换
    success = convert(
        pytorch_weight_path=args.pytorch_weight,
        output_path=args.output,
        verify=args.verify,
    )
    
    sys.exit(0 if success else 1)


# ============================================================
# 测试代码
# ============================================================

def test_conversion():
    """测试转换功能（不需要实际的权重文件）"""
    print("测试权重转换功能...")
    
    # 创建模拟的 PyTorch 权重
    mock_weights = {
        "backbone.layer1.weight": np.random.randn(64, 3, 7, 7).astype(np.float32),
        "backbone.layer1.bias": np.random.randn(64).astype(np.float32),
        "transformer.encoder.weight": np.random.randn(256, 256).astype(np.float32),
        "transformer.decoder.weight": np.random.randn(256, 256).astype(np.float32),
        "head.class_embed.weight": np.random.randn(256, 256).astype(np.float32),
        "head.bbox_embed.weight": np.random.randn(4, 256).astype(np.float32),
    }
    
    print_weight_info(mock_weights, "模拟权重")
    
    # 测试名称转换
    converted = convert_weight_names(mock_weights)
    print(f"\n名称转换后: {len(converted)} 个权重")
    
    # 测试保存和加载
    test_path = "test_weights_temp.pkl"
    save_jittor_weights(converted, test_path)
    loaded = load_jittor_weights(test_path)
    
    # 验证
    passed, _ = verify_conversion(converted, loaded)
    
    # 清理
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("\n测试完成！" if passed else "\n测试失败！")
    
    return passed


if __name__ == "__main__":
    # 如果没有参数，运行测试
    if len(sys.argv) == 1:
        print("未提供参数，运行测试模式...")
        print("使用 --help 查看帮助信息\n")
        test_conversion()
    else:
        main()
