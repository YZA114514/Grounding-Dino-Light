# ------------------------------------------------------------------------
# Grounding DINO - Jittor Implementation
# 检查实现完成度
# ------------------------------------------------------------------------
"""
检查 Jittor 实现的完成度

使用方法:
    python scripts/check_implementation_status.py
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================
# 需要检查的模块
# ============================================================

MODULES_TO_CHECK = {
    # 成员A - 核心模型架构
    "backbone": {
        "file": "jittor_implementation/models/backbone/swin_transformer.py",
        "class": "SwinTransformer",
        "member": "A",
        "priority": "P0",
        "description": "Swin Transformer Backbone (图像编码器)"
    },
    "ms_deform_attn": {
        "file": "jittor_implementation/models/attention/ms_deform_attn.py",
        "class": "MSDeformAttn",
        "member": "A",
        "priority": "P0",
        "description": "Multi-Scale Deformable Attention (最复杂)"
    },
    "transformer_encoder": {
        "file": "jittor_implementation/models/transformer/encoder.py",
        "class": "TransformerEncoder",
        "member": "A",
        "priority": "P0",
        "description": "Transformer Encoder"
    },
    "transformer_decoder": {
        "file": "jittor_implementation/models/transformer/decoder.py",
        "class": "TransformerDecoder",
        "member": "A",
        "priority": "P0",
        "description": "Transformer Decoder"
    },
    "dino_head": {
        "file": "jittor_implementation/models/head/dino_head.py",
        "class": "DINOHead",
        "member": "A",
        "priority": "P0",
        "description": "DINO Detection Head"
    },
    "groundingdino": {
        "file": "jittor_implementation/models/groundingdino.py",
        "class": "GroundingDINO",
        "member": "A",
        "priority": "P0",
        "description": "完整的 GroundingDINO 模型"
    },
    
    # 成员B - 数据处理和评估
    "dataset": {
        "file": "jittor_implementation/data/dataset.py",
        "class": "LVISDataset",
        "member": "B",
        "priority": "P0",
        "description": "LVIS 数据集加载器"
    },
    "transforms": {
        "file": "jittor_implementation/data/transforms.py",
        "class": "build_transforms",
        "member": "B",
        "priority": "P0",
        "description": "数据预处理"
    },
    "focal_loss": {
        "file": "jittor_implementation/losses/focal_loss.py",
        "class": "FocalLoss",
        "member": "B",
        "priority": "P1",
        "description": "Focal Loss"
    },
    "giou_loss": {
        "file": "jittor_implementation/losses/giou_loss.py",
        "class": "GIoULoss",
        "member": "B",
        "priority": "P1",
        "description": "GIoU Loss"
    },
    "grounding_loss": {
        "file": "jittor_implementation/losses/grounding_loss.py",
        "class": "GroundingLoss",
        "member": "B",
        "priority": "P1",
        "description": "Grounding DINO 总损失"
    },
    "lvis_evaluator": {
        "file": "jittor_implementation/eval/lvis_evaluator.py",
        "class": "LVISEvaluator",
        "member": "B",
        "priority": "P1",
        "description": "LVIS 评估器"
    },
    
    # 成员C - 文本编码和训练
    "bert_wrapper": {
        "file": "jittor_implementation/models/text_encoder/bert_wrapper.py",
        "class": "BERTWrapper",
        "member": "C",
        "priority": "P0",
        "description": "BERT 文本编码器包装"
    },
    "text_processor": {
        "file": "jittor_implementation/models/text_encoder/text_processor.py",
        "class": "TextProcessor",
        "member": "C",
        "priority": "P0",
        "description": "文本处理器"
    },
    "feature_fusion": {
        "file": "jittor_implementation/models/fusion/feature_fusion.py",
        "class": "FeatureFusion",
        "member": "C",
        "priority": "P0",
        "description": "特征融合模块"
    },
    "language_guided_query": {
        "file": "jittor_implementation/models/query/language_guided_query.py",
        "class": "LanguageGuidedQuery",
        "member": "C",
        "priority": "P1",
        "description": "语言引导 Query 生成"
    },
    "trainer": {
        "file": "jittor_implementation/train/trainer.py",
        "class": "Trainer",
        "member": "C",
        "priority": "P1",
        "description": "训练器"
    },
}


# ============================================================
# 检查函数
# ============================================================

def check_file_exists(file_path: str) -> bool:
    """检查文件是否存在"""
    full_path = project_root / file_path
    return full_path.exists()


def check_file_has_content(file_path: str, min_lines: int = 10) -> bool:
    """检查文件是否有实际内容（不只是占位符）"""
    full_path = project_root / file_path
    if not full_path.exists():
        return False
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 过滤空行和注释
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            return len(code_lines) >= min_lines
    except:
        return False


def check_class_implemented(file_path: str, class_name: str) -> Tuple[bool, str]:
    """
    检查类是否实现
    
    Returns:
        (是否实现, 错误信息)
    """
    full_path = project_root / file_path
    
    if not full_path.exists():
        return False, "文件不存在"
    
    try:
        # 读取文件内容
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否有类定义或函数定义
        is_class = f"class {class_name}" in content
        is_function = f"def {class_name}" in content
        
        if not is_class and not is_function:
            return False, "类/函数未定义"
        
        # 如果是函数（如 build_transforms），只要定义存在且有内容就算实现
        if is_function:
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            code_lines = [l for l in lines if not l.startswith('#')]
            if len(code_lines) >= 10:
                return True, "已实现"
            return False, "可能是占位符（代码行数太少）"
        
        # 对于类，检查是否是 nn.Module 子类
        is_nn_module = "nn.Module" in content or "(Module)" in content
        
        # 对于非 nn.Module 的类，不需要 execute/forward 方法
        # 而是检查其他重要方法
        non_module_classes = ["Dataset", "Evaluator", "Trainer", "Sampler", "Compose"]
        is_non_module = any(nc in class_name or f"({nc})" in content or f": {nc}" in content 
                           for nc in non_module_classes)
        
        if is_non_module or not is_nn_module:
            # 非神经网络模块，检查是否有实际功能方法
            has_methods = ("def __getitem__" in content or 
                          "def __call__" in content or 
                          "def evaluate" in content or 
                          "def train" in content or
                          "def __iter__" in content)
            
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            code_lines = [l for l in lines if not l.startswith('#')]
            
            if len(code_lines) >= 20 and (has_methods or len(code_lines) >= 50):
                return True, "已实现"
            elif len(code_lines) < 10:
                return False, "可能是占位符（代码行数太少）"
        else:
            # 对于 nn.Module，检查 execute 或 forward 方法
            if "def execute" not in content and "def forward" not in content:
                return False, "缺少 execute/forward 方法"
        
        # 检查是否只是占位符
        placeholder_keywords = [
            "NotImplementedError",
            "raise NotImplementedError",
            "# TODO",
            "# Placeholder"
        ]
        
        # 如果只有很少的代码行，可能是占位符
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        code_lines = [l for l in lines if not l.startswith('#')]
        
        if len(code_lines) < 10:
            return False, "可能是占位符（代码行数太少）"
        
        # 检查是否有太多占位符关键词
        placeholder_count = sum(1 for kw in placeholder_keywords if kw in content)
        if placeholder_count > 2 and len(code_lines) < 50:
            return False, "可能包含太多占位符"
        
        return True, "已实现"
        
    except Exception as e:
        return False, f"检查时出错: {str(e)}"


def check_importable(module_path: str, class_name: str) -> Tuple[bool, str]:
    """
    尝试导入模块并检查类是否可用
    
    Returns:
        (是否可导入, 错误信息)
    """
    try:
        # 转换为导入路径
        import_path = module_path.replace('/', '.').replace('.py', '')
        
        # 尝试导入
        spec = importlib.util.spec_from_file_location(
            import_path,
            project_root / module_path
        )
        
        if spec is None:
            return False, "无法创建模块规范"
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 检查类是否存在
        if not hasattr(module, class_name):
            return False, f"模块中不存在类 {class_name}"
        
        cls = getattr(module, class_name)
        
        # 检查是否可以实例化（不实际实例化，只检查）
        if not callable(cls):
            return False, f"{class_name} 不可调用"
        
        return True, "可导入"
        
    except Exception as e:
        return False, f"导入失败: {str(e)}"


# ============================================================
# 主检查函数
# ============================================================

def check_all_modules() -> Dict:
    """检查所有模块"""
    results = {
        "implemented": [],
        "partial": [],
        "missing": [],
        "by_member": {"A": [], "B": [], "C": []},
        "by_priority": {"P0": [], "P1": [], "P2": []},
    }
    
    print("=" * 80)
    print("Grounding DINO Jittor 实现完成度检查")
    print("=" * 80)
    print()
    
    for module_name, info in MODULES_TO_CHECK.items():
        file_path = info["file"]
        class_name = info["class"]
        member = info["member"]
        priority = info["priority"]
        description = info["description"]
        
        # 检查文件
        file_exists = check_file_exists(file_path)
        has_content = check_file_has_content(file_path) if file_exists else False
        
        # 检查类实现
        is_implemented, error_msg = check_class_implemented(file_path, class_name) if file_exists else (False, "文件不存在")
        
        # 尝试导入（可选，可能因为依赖问题失败）
        is_importable, import_error = (False, "未尝试") if not is_implemented else check_importable(file_path, class_name)
        
        # 判断状态
        if is_implemented and has_content:
            status = "✓ 已实现"
            results["implemented"].append(module_name)
        elif file_exists and has_content:
            status = "⚠ 部分实现"
            results["partial"].append(module_name)
        else:
            status = "✗ 未实现"
            results["missing"].append(module_name)
        
        # 按成员分类
        results["by_member"][member].append({
            "name": module_name,
            "status": status,
            "description": description
        })
        
        # 按优先级分类
        results["by_priority"][priority].append({
            "name": module_name,
            "status": status,
            "member": member
        })
        
        # 打印结果
        print(f"{status:12} | {member:2} | {priority:3} | {module_name:25} | {description}")
        if not is_implemented:
            print(f"             └─ {error_msg}")
    
    return results


def print_summary(results: Dict):
    """打印总结"""
    print()
    print("=" * 80)
    print("总结")
    print("=" * 80)
    
    total = len(MODULES_TO_CHECK)
    implemented = len(results["implemented"])
    partial = len(results["partial"])
    missing = total - implemented - partial
    
    print(f"\n总体进度:")
    print(f"  已实现: {implemented}/{total} ({implemented/total*100:.1f}%)")
    print(f"  部分实现: {partial}/{total} ({partial/total*100:.1f}%)")
    print(f"  未实现: {missing}/{total} ({missing/total*100:.1f}%)")
    
    print(f"\n按成员分工:")
    for member in ["A", "B", "C"]:
        modules = results["by_member"][member]
        implemented_count = sum(1 for m in modules if "✓" in m["status"])
        total_count = len(modules)
        print(f"  成员{member}: {implemented_count}/{total_count} ({implemented_count/total_count*100:.1f}%)")
    
    print(f"\n按优先级:")
    for priority in ["P0", "P1", "P2"]:
        modules = results["by_priority"][priority]
        implemented_count = sum(1 for m in modules if "✓" in m["status"])
        total_count = len(modules)
        if total_count > 0:
            print(f"  {priority}: {implemented_count}/{total_count} ({implemented_count/total_count*100:.1f}%)")
    
    # 列出未实现的 P0 模块
    print(f"\n⚠️  未实现的 P0 模块（高优先级）:")
    p0_missing = [m for m in results["by_priority"]["P0"] if "✗" in m["status"]]
    if p0_missing:
        for m in p0_missing:
            print(f"    - {m['name']} (成员{m['member']})")
    else:
        print("    无（所有 P0 模块已实现）")


def print_integration_guide():
    """打印集成验证指南"""
    print()
    print("=" * 80)
    print("代码合并和验证指南")
    print("=" * 80)
    
    print("""
1. 代码合并步骤：

   a) 确保所有成员的最新代码已推送到仓库
      git pull origin main

   b) 检查是否有冲突
      git status

   c) 如果有冲突，解决后提交
      git add .
      git commit -m "Resolve merge conflicts"
      git push origin main

2. 验证完整模型：

   a) 检查所有模块是否可以导入
      python -c "from jittor_implementation.models import GroundingDINO"

   b) 测试模型实例化
      python scripts/test_model_integration.py

   c) 测试前向传播（使用虚拟数据）
      python scripts/test_forward_pass.py

   d) 测试权重加载
      python scripts/test_weight_loading.py

3. 端到端测试：

   a) 数据加载测试
      python scripts/test_data_loading.py

   b) 训练循环测试（小数据集）
      python scripts/test_training_loop.py

   c) 评估测试
      python scripts/test_evaluation.py

4. 常见问题：

   - 模块导入错误：检查 __init__.py 是否正确导出
   - 形状不匹配：检查各模块的输入输出接口
   - 权重加载失败：检查权重名称映射
   - CUDA 错误：检查 Jittor 是否正确安装 GPU 支持
""")


# ============================================================
# 主函数
# ============================================================

def main():
    results = check_all_modules()
    print_summary(results)
    print_integration_guide()


if __name__ == "__main__":
    main()

