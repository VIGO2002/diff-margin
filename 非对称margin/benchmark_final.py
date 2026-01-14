import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, accuracy_score
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import random

# ================= ⚙️ 配置区域 (请根据实际情况微调) =================

# 1. 真实图片基准路径 (必须设置正确)
# 建议使用 CNNDetection 的 val 文件夹，或者 train/0_real
REAL_ROOT = "/root/autodl-tmp/datasets/CNNDetection/val"

# 2. 权重文件夹路径
CHECKPOINT_DIR = "./checkpoints/effort_universal_repro"

# 3. 要测试的 Epoch 列表 (建议测 1-9)
EPOCHS_TO_TEST = list(range(1, 10))

# 4. 每个数据集最大测试数量 (500代表: 500真 + 500假 = 1000张)
# 调大这个数值结果更准，但速度会变慢
MAX_SAMPLES_PER_CLASS = 500

# 5. 数据集路径映射 (根据你提供的 ls 结果配置)
FAKE_DATASETS = {
    # --- GAN 家族 (CNNDetection) ---
    "ProGAN": "/root/autodl-tmp/datasets/CNNDetection/progan",
    "CycleGAN": "/root/autodl-tmp/datasets/CNNDetection/cyclegan",
    "BigGAN": "/root/autodl-tmp/datasets/CNNDetection/biggan",
    "StyleGAN": "/root/autodl-tmp/datasets/CNNDetection/stylegan",
    "StyleGAN2": "/root/autodl-tmp/datasets/CNNDetection/stylegan2",
    "GauGAN": "/root/autodl-tmp/datasets/CNNDetection/gaugan",
    "StarGAN": "/root/autodl-tmp/datasets/CNNDetection/stargan",
    "DeepFake": "/root/autodl-tmp/datasets/CNNDetection/deepfake",

    # --- Diffusion 家族 (Diffusion) ---
    "LDM_200": "/root/autodl-tmp/datasets/Diffusion/ldm_200",
    "LDM_200_cfg": "/root/autodl-tmp/datasets/Diffusion/ldm_200_cfg",
    "LDM_100": "/root/autodl-tmp/datasets/Diffusion/ldm_100",
    "Glide_100_27": "/root/autodl-tmp/datasets/Diffusion/glide_100_27",
    "Glide_50_27": "/root/autodl-tmp/datasets/Diffusion/glide_50_27",
    "Glide_100_10": "/root/autodl-tmp/datasets/Diffusion/glide_100_10",
    "DALLE": "/root/autodl-tmp/datasets/Diffusion/dalle",
    
    # --- 高难度 ---
    "Guided": "/root/autodl-tmp/datasets/Diffusion/guided",
}

# ===================================================================

class BinaryEvalDataset(Dataset):
    def __init__(self, real_root, fake_root, transform=None, max_samples=500):
        self.transform = transform
        self.samples = []
        
        # --- 1. 加载真图 (Label 0) ---
        real_imgs = []
        if os.path.exists(real_root):
            for root, _, files in os.walk(real_root):
                # 🛡️ 过滤：如果在真图目录里发现了 'fake' 字样的文件夹，跳过
                if 'fake' in root.lower():
                    continue
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.webp')):
                        real_imgs.append((os.path.join(root, file), 0))
        
        # --- 2. 加载假图 (Label 1) ---
        fake_imgs = []
        if os.path.exists(fake_root):
            for root, _, files in os.walk(fake_root):
                # 🛡️ 核心修复：如果在假图目录里发现了 '0_real' 或 'real'，必须跳过！
                # 之前就是这里把真图当假图读了，导致准确率只有 50%
                if '0_real' in root or 'real' in root.lower():
                    continue
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.webp')):
                        fake_imgs.append((os.path.join(root, file), 1))

        # --- 3. 数据采样与平衡 ---
        # 打乱顺序
        random.shuffle(real_imgs)
        random.shuffle(fake_imgs)
        
        # 截断到最大数量
        if max_samples:
            real_imgs = real_imgs[:max_samples]
            fake_imgs = fake_imgs[:max_samples]
        
        # 强制数量平衡 (取最小值)，确保真假比例 1:1
        min_len = min(len(real_imgs), len(fake_imgs))
        
        if min_len == 0:
            print(f"⚠️  [Warning] Dataset empty or imbalanced! Real: {len(real_imgs)}, Fake: {len(fake_imgs)}")
            self.samples = []
        else:
            self.samples = real_imgs[:min_len] + fake_imgs[:min_len]
            print(f"    ✅ Loaded: {min_len} Real + {min_len} Fake = {len(self.samples)} Total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            # 遇到坏图返回黑图，防止程序中断
            return torch.zeros(3, 224, 224), label

def load_model(epoch, device):
    """加载模型并处理权重键名不匹配问题"""
    from models.clip_models import ClipModel
    try:
        # 初始化模型 (参数必须与训练时一致)
        model = ClipModel(
            name='openai/clip-vit-large-patch14', 
            num_classes=1, 
            fix_backbone=True, 
            use_svd=True, 
            svd_rank_ratio=0.25
        )
        
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
        if not os.path.exists(ckpt_path):
            print(f"❌ Checkpoint not found: {ckpt_path}")
            return None
            
        print(f"⚡️ Loading Epoch {epoch} weights...")
        
        # 加载权重
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # 兼容不同的保存格式
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 去除 'module.' 前缀 (如果是 DataParallel 保存的)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def main():
    # 设置随机种子确保可复现
    random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    
    # CLIP 标准预处理
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    results_data = []

    # --- 外层循环：遍历所有 Epoch ---
    for epoch in EPOCHS_TO_TEST:
        print(f"\n{'='*20} Testing Epoch {epoch} {'='*20}")
        model = load_model(epoch, device)
        if model is None: continue
        
        epoch_result = {'Epoch': epoch}
        
        # --- 内层循环：遍历所有数据集 ---
        for ds_name, fake_path in FAKE_DATASETS.items():
            print(f"📂 Dataset: {ds_name}")
            
            # 初始化数据集
            dataset = BinaryEvalDataset(REAL_ROOT, fake_path, transform=transform, max_samples=MAX_SAMPLES_PER_CLASS)
            
            if len(dataset) == 0:
                epoch_result[ds_name] = 0.0
                continue
            
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
            
            y_true = []
            y_scores = []
            
            with torch.no_grad():
                for imgs, labels in tqdm(loader, leave=False, desc=f"Evaluating {ds_name}"):
                    imgs = imgs.to(device)
                    # 前向传播
                    logits = model(imgs)
                    # 计算概率: Softmax 后取 Fake 类 (index 1) 的概率
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    
                    y_true.extend(labels.cpu().numpy())
                    y_scores.extend(probs.cpu().numpy())
            
            # 计算指标
            ap = average_precision_score(y_true, y_scores) * 100
            acc = accuracy_score(y_true, [1 if p > 0.5 else 0 for p in y_scores]) * 100
            
            print(f"   👉 AP: {ap:.2f}% | Acc: {acc:.2f}%")
            epoch_result[ds_name] = ap
        
        results_data.append(epoch_result)

    # --- 保存结果 ---
    print("\n" + "="*50)
    print("🏆 FINAL BENCHMARK RESULTS (AP %)")
    print("="*50)
    
    df = pd.DataFrame(results_data)
    df = df.set_index('Epoch')
    print(df)
    
    csv_filename = "benchmark_final_results.csv"
    df.to_csv(csv_filename)
    print(f"\n💾 Results saved to {csv_filename}")

if __name__ == "__main__":
    main()