import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, accuracy_score
from models.trainer import Trainer
from options.train_options import TrainOptions
import numpy as np
from tqdm import tqdm
import pandas as pd

# ==============================================================================
# 🌍 数据集全家桶路径配置
# ==============================================================================
DATASET_ROOTS = {
    # --- 1. CNN-based (Wang2020) ---
    'ProGAN': '/root/autodl-tmp/datasets/CNNDetection/test', # 或者是 progan
    'CycleGAN': '/root/autodl-tmp/datasets/CNNDetection/cyclegan',
    'BigGAN': '/root/autodl-tmp/datasets/CNNDetection/biggan',
    'StyleGAN': '/root/autodl-tmp/datasets/CNNDetection/stylegan',
    'StyleGAN2': '/root/autodl-tmp/datasets/CNNDetection/stylegan2',
    'GauGAN': '/root/autodl-tmp/datasets/CNNDetection/gaugan',
    'StarGAN': '/root/autodl-tmp/datasets/CNNDetection/stargan',
    'DeepFake': '/root/autodl-tmp/datasets/CNNDetection/deepfake',
    
    # --- 2. Diffusion-based ---
    'LDM_200': '/root/autodl-tmp/datasets/Diffusion/ldm_200',
    'LDM_200_cfg': '/root/autodl-tmp/datasets/Diffusion/ldm_200_cfg',
    'LDM_100': '/root/autodl-tmp/datasets/Diffusion/ldm_100',
    'Glide_100_27': '/root/autodl-tmp/datasets/Diffusion/glide_100_27',
    'Glide_50_27': '/root/autodl-tmp/datasets/Diffusion/glide_50_27',
    'Glide_100_10': '/root/autodl-tmp/datasets/Diffusion/glide_100_10',
    'DALLE': '/root/autodl-tmp/datasets/Diffusion/dalle',
    'Guided': '/root/autodl-tmp/datasets/Diffusion/guided',
    
    # --- 3. Extra Difficult Test Sets (Key for Selection) ---
    'SAN': '/root/autodl-tmp/datasets/Extra_Test/extracted_test/san',
    'CRN': '/root/autodl-tmp/datasets/Extra_Test/extracted_test/crn',
    'IMLE': '/root/autodl-tmp/datasets/Extra_Test/extracted_test/imle',
    'SITD': '/root/autodl-tmp/datasets/Extra_Test/extracted_test/sitd',
}

def test_epoch_on_dataset(model, dataset_name, dataset_path):
    # Data Transform
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    # 路径回退逻辑
    real_path = dataset_path
    if not os.path.exists(real_path):
        # 尝试一些常见的子目录结构
        if os.path.exists(os.path.join(dataset_path, 'val')): real_path = os.path.join(dataset_path, 'val')
        elif os.path.exists(os.path.join(dataset_path, 'test')): real_path = os.path.join(dataset_path, 'test')
    
    if not os.path.exists(real_path):
        # print(f"⚠️ Skip {dataset_name}: Path not found ({real_path})")
        return None, None

    try:
        dataset = datasets.ImageFolder(root=real_path, transform=val_transform)
        if len(dataset) == 0: return None, None
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    except:
        return None, None

    y_true, y_pred = [], []
    model.model.cuda()
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            model.set_input(data)
            model.test()
            pred = model.output
            
            if pred.shape[1] == 1:
                prob = torch.sigmoid(pred).cpu().numpy().flatten()
            else:
                prob = torch.softmax(pred, dim=1)[:, 1].cpu().numpy()
            
            # 强制二值化标签
            batch_targets = data[1].cpu().numpy()
            batch_targets = (batch_targets >= 1).astype(int)
            
            y_true.extend(batch_targets)
            y_pred.extend(prob)
            
    if len(np.unique(y_true)) < 2: return None, None
    return average_precision_score(y_true, y_pred), accuracy_score(y_true, [1 if p>0.5 else 0 for p in y_pred])

if __name__ == "__main__":
    # --- Init Config ---
    opt = TrainOptions().parse(print_options=False)
    opt.isTrain = False
    opt.gpu_ids = [0]
    opt.name = 'run_v4_final' # 你的实验名
    opt.checkpoints_dir = './checkpoints'
    opt.arch = 'CLIP:ViT-L/14_svd' 
    opt.fix_backbone = True
    opt.use_svd = True
    
    # --- Find Epochs ---
    ckpt_dir = os.path.join(opt.checkpoints_dir, opt.name)
    files = os.listdir(ckpt_dir)
    epochs = []
    for f in files:
        if f.startswith('model_epoch_') and f.endswith('.pth') and 'init' not in f:
            try: epochs.append(int(f.split('_')[-1].split('.')[0]))
            except: pass
    epochs.sort()
    
    print(f"🚀 Found Epochs: {epochs}")
    print(f"🌍 Testing on {len(DATASET_ROOTS)} datasets...")

    # --- Main Loop ---
    # 结果存储: {dataset: {epoch: score}}
    final_results = {k: {} for k in DATASET_ROOTS.keys()}
    
    try:
        model = Trainer(opt)
        model.eval()
    except Exception as e:
        print(f"Init failed: {e}")
        exit()

    for ep in epochs:
        print(f"\n{'='*20} Testing Epoch {ep} {'='*20}")
        
        # Load Weights
        ckpt_path = os.path.join(ckpt_dir, f'model_epoch_{ep}.pth')
        checkpoint = torch.load(ckpt_path, map_location='cuda:0')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        if hasattr(model.model, "module"): model.model.module.load_state_dict(state_dict)
        else: model.model.load_state_dict(state_dict)
        
        # Test all datasets
        for ds_name, ds_path in tqdm(DATASET_ROOTS.items(), leave=False):
            mAP, acc = test_epoch_on_dataset(model, ds_name, ds_path)
            if mAP is not None:
                final_results[ds_name][ep] = mAP * 100 # 转百分比
                # print(f"  {ds_name}: {mAP*100:.2f}%")
            else:
                final_results[ds_name][ep] = 0.0

    # --- Generate Report ---
    print("\n\n" + "="*50)
    print("🏆 GRAND FINAL RESULTS (mAP %)")
    print("="*50)
    
    df = pd.DataFrame(final_results).T # 转置：行是数据集，列是Epoch
    df = df.sort_index() # 按数据集名字排序
    
    # 打印表格
    print(df.to_string(float_format="%.2f"))
    
    # 计算平均分
    print("\n" + "-"*30)
    print("📊 Average mAP per Epoch")
    print("-" * 30)
    avg_scores = df.mean(axis=0)
    print(avg_scores.to_string(float_format="%.2f"))
    
    best_ep = avg_scores.idxmax()
    print(f"\n🥇 Best Epoch (Overall): {best_ep} (mAP: {avg_scores[best_ep]:.2f}%)")
    
    # 保存结果
    df.to_csv("benchmark_all_epochs_v4.csv")
    print("\n💾 Saved to benchmark_all_epochs_v4.csv")