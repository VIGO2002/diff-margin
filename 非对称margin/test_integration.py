import torch
import sys
import os

# 设置环境变量，模拟 options 参数
sys.argv = ['test_integration.py', '--arch', 'CLIP:ViT-L/14_svd', '--fix_backbone', '--use_svd']

def test_all_modes():
    print("\n🧪 开始 Dual-EBM 集成测试...")
    # 【修改】引入正确的类名 ClipModel
    from models.clip_models import ClipModel
    
    # 1. 创建模型
    print("1. 初始化模型 (CLIP ViT-L/14 + SVD)...")
    model = ClipModel(
        "openai/clip-vit-large-patch14",
        use_svd=True,
        noise_std=0.01,
        svd_rank_ratio=0.25
    )
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    # 2. 测试训练模式
    print("\n2. 测试训练模式 (Forward)...")
    model.train()
    output = model(x)
    assert isinstance(output, tuple) and len(output) == 5, "❌ 训练模式应返回 5 个值"
    logits, e_real, e_fake, e_real_noisy, e_fake_noisy = output
    print(f"   ✅ 输出形状检查: Logits {logits.shape}, Energy {e_real.shape}")
    
    # 3. 测试验证模式
    print("\n3. 测试验证模式 (Eval)...")
    model.eval()
    
    # 3.1 默认 (只返回 Logits)
    logits_only = model(x)
    assert isinstance(logits_only, torch.Tensor), "❌ 验证模式默认应返回 Tensor"
    print(f"   ✅ 默认验证通过")
    
    # 3.2 强制返回能量
    logits_energy = model(x, return_energy=True)
    assert isinstance(logits_energy, tuple) and len(logits_energy) == 3, "❌ 验证模式(return_energy=True)应返回 3 个值"
    print(f"   ✅ 能量验证通过")

    # 4. 模拟 Trainer 反向传播
    print("\n4. 测试反向传播 (Backward)...")
    model.train()
    logits, e_real, e_fake, e_real_n, e_fake_n = model(x)
    
    loss = logits.mean() + e_real.mean() + e_real_n.mean()
    loss.backward()
    
    # 检查 SVD 残差是否有梯度
    has_grad = False
    for name, param in model.named_parameters():
        if 'S_residual' in name and param.grad is not None:
            has_grad = True
            print(f"   ✅ 梯度检查: {name} 有梯度 (Mean: {param.grad.abs().mean():.6f})")
            break
    
    if not has_grad:
        print("   ❌ 警告: SVD Residual 没有接收到梯度！")
    else:
        print("   🎉 所有测试通过！可以开始训练！")

if __name__ == "__main__":
    test_all_modes()