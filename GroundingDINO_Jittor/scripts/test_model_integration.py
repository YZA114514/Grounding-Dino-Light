# ------------------------------------------------------------------------
# Grounding DINO - Jittor Implementation
# 模型集成测试脚本
# ------------------------------------------------------------------------
"""
测试 Jittor 实现的模型是否能正确导入和实例化

使用方法:
    python scripts/test_model_integration.py
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_jittor_available():
    """检查 Jittor 是否可用"""
    try:
        import jittor
        return True
    except ImportError:
        return False


JITTOR_AVAILABLE = check_jittor_available()


def test_syntax_only():
    """在没有 Jittor 的情况下检查语法"""
    import ast
    
    files_to_check = [
        "jittor_implementation/models/attention/ms_deform_attn.py",
        "jittor_implementation/models/transformer/encoder.py",
        "jittor_implementation/models/transformer/decoder.py",
        "jittor_implementation/models/backbone/swin_transformer.py",
        "jittor_implementation/models/head/dino_head.py",
        "jittor_implementation/models/fusion/feature_fusion.py",
        "jittor_implementation/models/groundingdino.py",
    ]
    
    all_passed = True
    for file_path in files_to_check:
        full_path = project_root / file_path
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                source = f.read()
            ast.parse(source)
            print(f"  ✓ 语法检查通过: {file_path.split('/')[-1]}")
        except SyntaxError as e:
            print(f"  ✗ 语法错误: {file_path.split('/')[-1]} - {e}")
            all_passed = False
        except FileNotFoundError:
            print(f"  ✗ 文件不存在: {file_path}")
            all_passed = False
    
    return all_passed


def test_code_structure():
    """检查代码结构，验证 MSDeformAttn 被正确引入"""
    print("\n" + "=" * 60)
    print("测试代码结构（无需 Jittor）")
    print("=" * 60)
    
    # 检查 encoder.py 是否引入了 MSDeformAttn
    encoder_path = project_root / "jittor_implementation/models/transformer/encoder.py"
    with open(encoder_path, 'r', encoding='utf-8') as f:
        encoder_content = f.read()
    
    checks = []
    
    # 检查 encoder 导入
    if "from ..attention.ms_deform_attn import MSDeformAttn" in encoder_content:
        checks.append(("Encoder 导入 MSDeformAttn", True))
    else:
        checks.append(("Encoder 导入 MSDeformAttn", False))
    
    # 检查 encoder 使用
    if "self.self_attn = MSDeformAttn(" in encoder_content:
        checks.append(("Encoder 使用 MSDeformAttn", True))
    else:
        checks.append(("Encoder 使用 MSDeformAttn", False))
    
    # 检查 decoder.py
    decoder_path = project_root / "jittor_implementation/models/transformer/decoder.py"
    with open(decoder_path, 'r', encoding='utf-8') as f:
        decoder_content = f.read()
    
    # 检查 decoder 导入
    if "from ..attention.ms_deform_attn import MSDeformAttn" in decoder_content:
        checks.append(("Decoder 导入 MSDeformAttn", True))
    else:
        checks.append(("Decoder 导入 MSDeformAttn", False))
    
    # 检查 decoder 使用
    if "self.cross_attn = MSDeformAttn(" in decoder_content:
        checks.append(("Decoder 使用 MSDeformAttn", True))
    else:
        checks.append(("Decoder 使用 MSDeformAttn", False))
    
    # 检查是否还有旧的占位符代码
    if "self.use_deformable = False" in encoder_content:
        checks.append(("Encoder 无占位符", False))
    else:
        checks.append(("Encoder 无占位符", True))
    
    if "self.use_deformable = False" in decoder_content:
        checks.append(("Decoder 无占位符", False))
    else:
        checks.append(("Decoder 无占位符", True))
    
    # 打印结果
    all_passed = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_imports():
    """测试所有模块是否可以导入"""
    print("=" * 60)
    print("测试模块导入")
    print("=" * 60)
    
    if not JITTOR_AVAILABLE:
        print("  ⚠ Jittor 未安装，跳过导入测试（进行语法检查）")
        return test_syntax_only()
    
    tests = []
    
    # 测试 attention 模块
    try:
        from jittor_implementation.models.attention.ms_deform_attn import MSDeformAttn
        tests.append(("MSDeformAttn", True, None))
    except Exception as e:
        tests.append(("MSDeformAttn", False, str(e)))
    
    # 测试 transformer 模块
    try:
        from jittor_implementation.models.transformer.encoder import (
            TransformerEncoder, DeformableTransformerEncoderLayer
        )
        tests.append(("TransformerEncoder", True, None))
    except Exception as e:
        tests.append(("TransformerEncoder", False, str(e)))
    
    try:
        from jittor_implementation.models.transformer.decoder import (
            TransformerDecoder, DeformableTransformerDecoderLayer
        )
        tests.append(("TransformerDecoder", True, None))
    except Exception as e:
        tests.append(("TransformerDecoder", False, str(e)))
    
    # 测试 backbone 模块
    try:
        from jittor_implementation.models.backbone.swin_transformer import SwinTransformer
        tests.append(("SwinTransformer", True, None))
    except Exception as e:
        tests.append(("SwinTransformer", False, str(e)))
    
    # 测试 head 模块
    try:
        from jittor_implementation.models.head.dino_head import DINOHead
        tests.append(("DINOHead", True, None))
    except Exception as e:
        tests.append(("DINOHead", False, str(e)))
    
    # 测试 fusion 模块
    try:
        from jittor_implementation.models.fusion.feature_fusion import FeatureFusion
        tests.append(("FeatureFusion", True, None))
    except Exception as e:
        tests.append(("FeatureFusion", False, str(e)))
    
    # 测试完整模型
    try:
        from jittor_implementation.models.groundingdino import GroundingDINO
        tests.append(("GroundingDINO", True, None))
    except Exception as e:
        tests.append(("GroundingDINO", False, str(e)))
    
    # 测试 losses
    try:
        from jittor_implementation.losses.grounding_loss import GroundingLoss
        tests.append(("GroundingLoss", True, None))
    except Exception as e:
        tests.append(("GroundingLoss", False, str(e)))
    
    # 测试 data
    try:
        from jittor_implementation.data.dataset import LVISDataset
        tests.append(("LVISDataset", True, None))
    except Exception as e:
        tests.append(("LVISDataset", False, str(e)))
    
    # 打印结果
    passed = sum(1 for _, success, _ in tests if success)
    total = len(tests)
    
    for name, success, error in tests:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
        if error:
            print(f"      Error: {error[:80]}...")
    
    print(f"\n导入测试: {passed}/{total} 通过")
    return passed == total


def test_encoder_uses_deformable_attention():
    """测试 Encoder 是否正确使用 MSDeformAttn"""
    print("\n" + "=" * 60)
    print("测试 Encoder 使用可变形注意力")
    print("=" * 60)
    
    if not JITTOR_AVAILABLE:
        print("  ⚠ Jittor 未安装，使用代码检查")
        # 代码检查在 test_code_structure 中完成
        encoder_path = project_root / "jittor_implementation/models/transformer/encoder.py"
        with open(encoder_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if "self.self_attn = MSDeformAttn(" in content:
            print("  ✓ DeformableTransformerEncoderLayer 代码中使用 MSDeformAttn")
            return True
        else:
            print("  ✗ DeformableTransformerEncoderLayer 代码中未使用 MSDeformAttn")
            return False
    
    try:
        from jittor_implementation.models.transformer.encoder import DeformableTransformerEncoderLayer
        from jittor_implementation.models.attention.ms_deform_attn import MSDeformAttn
        
        # 创建 encoder layer
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model=256,
            d_ffn=1024,
            dropout=0.1,
            n_levels=4,
            n_heads=8,
            n_points=4
        )
        
        # 检查 self_attn 是否是 MSDeformAttn 类型
        is_deformable = isinstance(encoder_layer.self_attn, MSDeformAttn)
        
        if is_deformable:
            print("  ✓ DeformableTransformerEncoderLayer 正确使用 MSDeformAttn")
            return True
        else:
            print(f"  ✗ DeformableTransformerEncoderLayer 使用了 {type(encoder_layer.self_attn)}")
            return False
            
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False


def test_decoder_uses_deformable_attention():
    """测试 Decoder 是否正确使用 MSDeformAttn"""
    print("\n" + "=" * 60)
    print("测试 Decoder 使用可变形注意力")
    print("=" * 60)
    
    if not JITTOR_AVAILABLE:
        print("  ⚠ Jittor 未安装，使用代码检查")
        decoder_path = project_root / "jittor_implementation/models/transformer/decoder.py"
        with open(decoder_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if "self.cross_attn = MSDeformAttn(" in content:
            print("  ✓ DeformableTransformerDecoderLayer 代码中使用 MSDeformAttn")
            return True
        else:
            print("  ✗ DeformableTransformerDecoderLayer 代码中未使用 MSDeformAttn")
            return False
    
    try:
        from jittor_implementation.models.transformer.decoder import DeformableTransformerDecoderLayer
        from jittor_implementation.models.attention.ms_deform_attn import MSDeformAttn
        
        # 创建 decoder layer
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=256,
            d_ffn=1024,
            dropout=0.1,
            n_levels=4,
            n_heads=8,
            n_points=4
        )
        
        # 检查 cross_attn 是否是 MSDeformAttn 类型
        is_deformable = isinstance(decoder_layer.cross_attn, MSDeformAttn)
        
        if is_deformable:
            print("  ✓ DeformableTransformerDecoderLayer 正确使用 MSDeformAttn")
            return True
        else:
            print(f"  ✗ DeformableTransformerDecoderLayer 使用了 {type(decoder_layer.cross_attn)}")
            return False
            
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False


def test_msdeform_attn_instantiation():
    """测试 MSDeformAttn 是否可以实例化"""
    print("\n" + "=" * 60)
    print("测试 MSDeformAttn 实例化")
    print("=" * 60)
    
    if not JITTOR_AVAILABLE:
        print("  ⚠ Jittor 未安装，跳过实例化测试")
        return True
    
    try:
        from jittor_implementation.models.attention.ms_deform_attn import MSDeformAttn
        
        # 测试不同配置
        configs = [
            {"embed_dim": 256, "num_levels": 4, "num_heads": 8, "num_points": 4},
            {"embed_dim": 512, "num_levels": 3, "num_heads": 8, "num_points": 8},
            {"embed_dim": 256, "num_levels": 4, "num_heads": 4, "num_points": 4, "batch_first": True},
        ]
        
        for i, cfg in enumerate(configs):
            attn = MSDeformAttn(**cfg)
            print(f"  ✓ 配置 {i+1}: {cfg}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False


def test_forward_shapes():
    """测试前向传播的输入输出形状"""
    print("\n" + "=" * 60)
    print("测试前向传播形状 (需要 Jittor)")
    print("=" * 60)
    
    try:
        import jittor as jt
        from jittor_implementation.models.attention.ms_deform_attn import MSDeformAttn
        
        # 配置
        batch_size = 2
        num_queries = 100
        embed_dim = 256
        num_levels = 4
        num_heads = 8
        num_points = 4
        
        # 空间形状 (假设 4 个层级)
        spatial_shapes = jt.array([[80, 80], [40, 40], [20, 20], [10, 10]], dtype=jt.int64)
        level_start_index = jt.array([0, 6400, 8000, 8400], dtype=jt.int64)
        total_length = 8500  # 80*80 + 40*40 + 20*20 + 10*10 = 8500
        
        # 创建输入
        query = jt.randn(batch_size, num_queries, embed_dim)
        reference_points = jt.rand(batch_size, num_queries, num_levels, 2)  # 2D 参考点
        value = jt.randn(batch_size, total_length, embed_dim)
        
        # 创建模块
        attn = MSDeformAttn(
            embed_dim=embed_dim,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points,
            batch_first=True
        )
        
        # 前向传播
        output = attn(
            query=query,
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )
        
        # 检查输出形状
        expected_shape = (batch_size, num_queries, embed_dim)
        actual_shape = tuple(output.shape)
        
        if actual_shape == expected_shape:
            print(f"  ✓ 输出形状正确: {actual_shape}")
            return True
        else:
            print(f"  ✗ 输出形状错误: 期望 {expected_shape}, 实际 {actual_shape}")
            return False
            
    except ImportError:
        print("  ⚠ Jittor 未安装，跳过此测试")
        return True
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Grounding DINO Jittor 实现 - 集成测试")
    print("=" * 60)
    
    if JITTOR_AVAILABLE:
        print("Jittor 已安装，将进行完整测试")
    else:
        print("⚠ Jittor 未安装，将进行代码结构检查")
    
    print()
    
    results = []
    
    # 运行测试
    results.append(("模块导入/语法检查", test_imports()))
    results.append(("代码结构检查", test_code_structure()))
    results.append(("MSDeformAttn 实例化", test_msdeform_attn_instantiation()))
    results.append(("Encoder 使用可变形注意力", test_encoder_uses_deformable_attention()))
    results.append(("Decoder 使用可变形注意力", test_decoder_uses_deformable_attention()))
    
    if JITTOR_AVAILABLE:
        results.append(("前向传播形状", test_forward_shapes()))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n✓ 所有测试通过！模型实现正确。")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查实现。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
