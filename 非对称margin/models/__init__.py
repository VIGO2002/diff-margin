from .clip_models import ClipModel
import sys

VALID_NAMES = {
    'CLIP:ViT-B/16_svd': 'openai/clip-vit-base-patch16',
    'CLIP:ViT-B/32_svd': 'openai/clip-vit-base-patch32',
    'CLIP:ViT-L/14_svd':'/root/autodl-tmp/pretrained_models/clip-vit-large-patch14',
    'SigLIP:ViT-L/16_256_svd': 'google/siglip-large-patch16-256',
    'BEiTv2:ViT-L/16_svd': 'microsoft/beit-v2-large-patch16-224',
}

def get_model(arch, opt):
    # arch 对应 train.py 里的 opt.arch
    if arch not in VALID_NAMES:
        print(f"❌ Error: Model architecture {arch} not implemented in VALID_NAMES.")
        sys.exit(1)

    print(f"Loading model: {arch} -> {VALID_NAMES[arch]}")
    
    # 【关键】必须把 opt 里的参数一个个传进去，否则 SVD 无法激活
    if arch.startswith("CLIP:"):
        return ClipModel(
            name=VALID_NAMES[arch],
            num_classes=1,
            fix_backbone=opt.fix_backbone, # 传入 fix_backbone
            use_svd=opt.use_svd,           # 传入 use_svd (关键!)
            noise_std=opt.noise_std,       # 传入 noise_std (关键!)
            svd_rank_ratio=0.25            # 可以写死或者从 opt 读
        )
    else:
        print(f"Architecture {arch} logic not implemented.")
        sys.exit(1)