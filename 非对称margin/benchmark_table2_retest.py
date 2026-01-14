import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import average_precision_score, accuracy_score
from PIL import Image
from tqdm import tqdm
import random

# ================= ⚙️ 配置区域 =================

# 1. 核心：原来你一直用的是这个真图目录！
REAL_ROOT = "/root/autodl-tmp/datasets/CNNDetection/val"

# 2. 你的最强模型 Epoch
EPOCH = 8
DEVICE = "cuda"

# 3. 🔥 修正：使用 0.001 计算 Acc
FIXED_THRESHOLD = 0.0010

# 4. 本地 CLIP 路径
CLIP_LOCAL_PATH = "/root/autodl-tmp/pretrained_models/clip-vit-large-patch14"

# 5. 你的数据集映射 (保持原样)
FAKE_DATASETS = {
    # --- GAN 家族 ---
    "ProGAN": "/root/autodl-tmp/datasets/CNNDetection/progan",
    "CycleGAN": "/root/autodl-tmp/datasets/CNNDetection/cyclegan",
    "BigGAN": "/root/autodl-tmp/datasets/CNNDetection/biggan",
    "StyleGAN": "/root/autodl-tmp/datasets/CNNDetection/stylegan",
    "StyleGAN2": "/root/autodl-tmp/datasets/CNNDetection/stylegan2",
    "GauGAN": "/root/autodl-tmp/datasets/CNNDetection/gaugan",
    "StarGAN": "/root/autodl-tmp/datasets/CNNDetection/stargan",
    "DeepFake": "/root/autodl-tmp/datasets/CNNDetection/deepfake",

    # --- Diffusion 家族 ---
    "LDM_200": "/root/autodl-tmp/datasets/Diffusion/ldm_200",
    "LDM_200_cfg": "/root/autodl-tmp/datasets/Diffusion/ldm_200_cfg",
    "LDM_100": "/root/autodl-tmp/datasets/Diffusion/ldm_100",
    "Glide_100_27": "/root/autodl-tmp/datasets/Diffusion/glide_100_27",
    "Glide_50_27": "/root/autodl-tmp/datasets/Diffusion/glide_50_27",
    "Glide_100_10": "/root/autodl-tmp/datasets/Diffusion/glide_100_10",
    "DALLE": "/root/autodl-tmp/datasets/Diffusion/dalle",
    
    # --- 高难度 ---
    "Guided": "/root/autodl-tmp/datasets/Diffusion/guided",
    # === 新增的泛化测试集 ===
    "SITD": "/root/autodl-tmp/datasets/Extra_Test/extracted_test/sitd",
    "SAN":  "/root/autodl-tmp/datasets/Extra_Test/extracted_test/san",
    "CRN":  "/root/autodl-tmp/datasets/Extra_Test/extracted_test/crn",
    "IMLE": "/root/autodl-tmp/datasets/Extra_Test/extracted_test/imle",
}

# 6. 采样数量 (500真+500假=1000，和你之前的Log一致)
MAX_SAMPLES_PER_CLASS = 500
CHECKPOINT_DIR = "./checkpoints/effort_universal_repro"

# ===============================================

class BinaryEvalDataset(Dataset):
    def __init__(self, real_root, fake_root, transform=None, max_samples=500):
        self.transform = transform
        self.samples = []
        
        # --- 1. 加载真图 (固定来自 CNNDetection/val) ---
        real_imgs = []
        if os.path.exists(real_root):
            for root, _, files in os.walk(real_root):
                if 'fake' in root.lower(): continue
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        real_imgs.append((os.path.join(root, file), 0))
        
        # --- 2. 加载假图 (来自各个数据集) ---
        fake_imgs = []
        if os.path.exists(fake_root):
            for root, _, files in os.walk(fake_root):
                # 排除可能存在的真图文件夹，只读假图
                if '0_real' in root or 'real' in root.lower(): continue
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        fake_imgs.append((os.path.join(root, file), 1))

        # --- 3. 采样 ---
        random.shuffle(real_imgs)
        random.shuffle(fake_imgs)
        
        if max_samples:
            real_imgs = real_imgs[:max_samples]
            fake_imgs = fake_imgs[:max_samples]
        
        min_len = min(len(real_imgs), len(fake_imgs))
        self.samples = real_imgs[:min_len] + fake_imgs[:min_len]
        
        # print(f"    Loaded: {min_len} Real + {min_len} Fake")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except:
            return torch.zeros(3, 224, 224), label

def load_model(epoch, device):
    from models.clip_models import ClipModel
    try:
        model = ClipModel(
            name=CLIP_LOCAL_PATH, 
            num_classes=1, 
            fix_backbone=True, 
            use_svd=True, 
            svd_rank_ratio=0.25
        )
        
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
        print(f"⚡️ Loading Epoch {epoch} weights...")
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k[7:] if k.startswith('module.') else k
            new_state_dict[k] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def main():
    torch.manual_seed(42)
    random.seed(42)
    
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    model = load_model(EPOCH, DEVICE)
    if model is None: return

    print(f"\n{'='*60}")
    print(f"🚀 FINAL BENCHMARK (Fixed Real Source)")
    print(f"🎯 Threshold: {FIXED_THRESHOLD}")
    print(f"📸 Real Source: {REAL_ROOT}")
    print(f"{'='*60}\n")
    
    print(f"{'Dataset':<20} | {'AP (%)':<10} | {'Acc (%)':<10}")
    print("-" * 46)

    for ds_name, fake_path in FAKE_DATASETS.items():
        if not os.path.exists(fake_path):
            print(f"⚠️  Skipping {ds_name}: Path not found")
            continue

        dataset = BinaryEvalDataset(REAL_ROOT, fake_path, transform=transform, max_samples=MAX_SAMPLES_PER_CLASS)
        
        if len(dataset) == 0:
            print(f"⚠️  Skipping {ds_name}: Empty")
            continue
            
        loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
        
        y_true, y_scores = [], []
        
        with torch.no_grad():
            for imgs, labels in tqdm(loader, leave=False, desc=ds_name):
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)[:, 1]
                
                y_true.extend(labels.cpu().numpy())
                y_scores.extend(probs.cpu().numpy())
        
        ap = average_precision_score(y_true, y_scores) * 100
        
        # 🔥 这里就是你要的修正：使用 0.001 计算 Acc
        y_pred = [1 if p > FIXED_THRESHOLD else 0 for p in y_scores]
        acc = accuracy_score(y_true, y_pred) * 100
        
        print(f"{ds_name:<20} | {ap:<10.2f} | {acc:<10.2f}")

if __name__ == "__main__":
    main()