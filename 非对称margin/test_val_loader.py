import os
from torchvision import datasets

# 定义 BinaryImageFolder
class BinaryImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, _ = self.samples[index]
        # 只要路径包含 '1_fake' 或 'fake' 就是 1，否则是 0
        if '1_fake' in path or 'fake' in path.lower():
            target = 1
        else:
            target = 0
        return path, target

# 指定路径
val_root = "/root/autodl-tmp/datasets/CNNDetection/val"

print(f"📂 正在检查路径: {val_root}")

try:
    # 加载数据集
    ds = BinaryImageFolder(root=val_root)
    print(f"✅ Indexed {len(ds)} images.")
    
    # 抽查统计
    real_count = 0
    fake_count = 0
    # 遍历整个数据集太慢，我们要么抽样，要么只检查前几千个
    # 这里我们每隔 50 张抽一张，速度会很快
    for i in range(0, len(ds), 50): 
        p, t = ds[i]
        if t == 1: fake_count += 1
        else: real_count += 1
        
    print(f"📊 Sampling stats (every 50th): Real={real_count}, Fake={fake_count}")
    
    if real_count > 0 and fake_count > 0:
        print("🎉 完美！验证集包含 Real 和 Fake 两种标签。")
    elif real_count > 0:
        print("⚠️ 警告：只检测到了 Real 标签！(可能是 Fake 图片路径里没有 'fake' 关键字)")
    elif fake_count > 0:
        print("⚠️ 警告：只检测到了 Fake 标签！")
    else:
        print("❌ 错误：没有检测到任何有效图片。")

except Exception as e:
    print(f"❌ 发生错误: {e}")
