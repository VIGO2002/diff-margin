import torch
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from benchmark_final import load_model, BinaryEvalDataset
from torch.utils.data import DataLoader
from torchvision import transforms  # <--- 必须导入这个
from tqdm import tqdm

# === 配置 ===
EPOCH = 8
DEVICE = "cuda"
CALIB_DATASET_PATH = "/root/autodl-tmp/datasets/CNNDetection/progan"
REAL_ROOT = "/root/autodl-tmp/datasets/CNNDetection/val"

TEST_DATASETS = {
    "Guided": "/root/autodl-tmp/datasets/Diffusion/guided",
    "LDM_200": "/root/autodl-tmp/datasets/Diffusion/ldm_200",
    "DeepFake": "/root/autodl-tmp/datasets/CNNDetection/deepfake"
}

def find_optimal_threshold(y_true, y_scores):
    best_acc = 0
    best_thresh = 0.5
    # 搜索最佳阈值
    for t in np.linspace(min(y_scores), max(y_scores), 1000):
        y_pred = [1 if s >= t else 0 for s in y_scores]
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh

def main():
    # 1. 定义预处理 (必须有！)
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    model = load_model(EPOCH, DEVICE)
    
    print(f"🔧 Step 1: Calibrating threshold on Source Domain ({CALIB_DATASET_PATH})...")
    
    # 2. 初始化校准数据集 (传入 transform)
    calib_ds = BinaryEvalDataset(
        REAL_ROOT, 
        CALIB_DATASET_PATH, 
        transform=transform,  # <--- 修复点：传入预处理
        max_samples=1000
    )
    
    if len(calib_ds) == 0:
        print("❌ Error: Calibration dataset is empty! Check path.")
        return

    calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=False)
    
    y_true_calib, y_scores_calib = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(calib_loader):
            imgs = imgs.to(DEVICE)
            # 计算概率
            probs = torch.softmax(model(imgs), dim=1)[:, 1]
            y_true_calib.extend(labels.cpu().numpy())
            y_scores_calib.extend(probs.cpu().numpy())
            
    # 3. 找到最佳阈值
    GLOBAL_THRESHOLD = find_optimal_threshold(y_true_calib, y_scores_calib)
    
    # 计算校准集上的 Acc
    calib_preds = [1 if s >= GLOBAL_THRESHOLD else 0 for s in y_scores_calib]
    calib_acc = accuracy_score(y_true_calib, calib_preds) * 100
    
    print(f"\n✅ Global Calibrated Threshold: {GLOBAL_THRESHOLD:.4f}")
    print(f"📊 Source Domain Acc (ProGAN): {calib_acc:.2f}%")
    print(f"(We will use this fixed threshold for ALL other datasets to ensure fairness)\n")
    
    # 4. Step 2: 用这个固定阈值测试其他
    print(f"🚀 Step 2: Testing Generalization with Global Threshold...")
    
    for name, path in TEST_DATASETS.items():
        # 这里也要传 transform
        ds = BinaryEvalDataset(REAL_ROOT, path, transform=transform, max_samples=500)
        if len(ds) == 0: continue
        ld = DataLoader(ds, batch_size=32, num_workers=4)
        
        y_true, y_scores = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(ld, desc=name, leave=False):
                imgs = imgs.to(DEVICE)
                probs = torch.softmax(model(imgs), dim=1)[:, 1]
                y_true.extend(labels.cpu().numpy())
                y_scores.extend(probs.cpu().numpy())
        
        # 使用 全局阈值 计算 Acc
        y_pred = [1 if s >= GLOBAL_THRESHOLD else 0 for s in y_scores]
        acc = accuracy_score(y_true, y_pred) * 100
        ap = average_precision_score(y_true, y_scores) * 100
        
        print(f"   👉 {name}: AP={ap:.2f}% | Acc={acc:.2f}% (Thresh={GLOBAL_THRESHOLD:.4f})")

if __name__ == "__main__":
    main()