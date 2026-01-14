import torch
import torch.nn as nn
from transformers import CLIPModel as HuggingFaceCLIPModel
import sys

class DualEnergyHead(nn.Module):
    def __init__(self, in_features, hidden_dim=512, init_noise_std=0.01):
        super(DualEnergyHead, self).__init__()
        
        # 1. 共享编码器 (Shared Encoder)
        self.shared_encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        )
        
        # 2. 独立的能量分支
        self.energy_real = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.energy_fake = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.noise_std = nn.Parameter(torch.tensor(init_noise_std))

    def forward(self, x, training=True):
        h = self.shared_encoder(x)
        
        if training:
            std = torch.clamp(self.noise_std, min=0.001, max=0.05)
            noise = torch.randn_like(h) * std
            h_noisy = h + noise
            
            e_real_noisy = self.energy_real(h_noisy)
            e_fake_noisy = self.energy_fake(h_noisy)
        else:
            e_real_noisy = None
            e_fake_noisy = None
        
        e_real = self.energy_real(h)
        e_fake = self.energy_fake(h)
        
        logits = torch.cat([-e_real, -e_fake], dim=1)
        
        if training:
            return logits, e_real, e_fake, e_real_noisy, e_fake_noisy
        else:
            return logits, e_real, e_fake

class ClipModel(nn.Module):
    def __init__(self, name, num_classes=1, fix_backbone=False, use_svd=False, noise_std=0.02, svd_rank_ratio=0.25):
        super(ClipModel, self).__init__()
        
        self.svd_rank_ratio = svd_rank_ratio
        print(f">>> Initializing CLIP with Dual-EBM (Shared Encoder, Noise std: {noise_std})")
        if use_svd:
            print(f">>> SVD Enabled with rank ratio: {self.svd_rank_ratio}")
            
        self.model = HuggingFaceCLIPModel.from_pretrained(name)
        
        # 【关键修复】直接在主模型上开启梯度检查点 (AttributeError 修复)
        # 这样会自动应用到 vision_model 和 text_model (如果有的话)
        self.model.gradient_checkpointing_enable()
        print(">>> Gradient Checkpointing Enabled (Memory Saved!)")
        
        if use_svd:
            print(f">>> Applying SVD Residual to CLIP backbone...")
            for layer in self.model.vision_model.encoder.layers:
                self._apply_svd_to_layer(layer.self_attn.k_proj)
                self._apply_svd_to_layer(layer.self_attn.v_proj)
                self._apply_svd_to_layer(layer.self_attn.q_proj)
                self._apply_svd_to_layer(layer.self_attn.out_proj)
        
        vision_dim = self.model.vision_model.config.hidden_size
        print(f">>> Vision Feature Dimension: {vision_dim} (ViT-L default is 1024)")
        
        self.fc = DualEnergyHead(vision_dim, init_noise_std=noise_std)

    def _apply_svd_to_layer(self, layer):
        weight = layer.weight.data
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        V = Vh.t() 
        
        rank = int(S.shape[0] * self.svd_rank_ratio)
        rank = max(1, rank)
        
        layer.weight_main = nn.Parameter(U[:, :rank] @ torch.diag(S[:rank]) @ V[:, :rank].t())
        layer.weight_main.requires_grad = False 
        
        layer.S_residual = nn.Parameter(S[rank:])
        layer.U_residual = nn.Parameter(U[:, rank:])
        layer.V_residual = nn.Parameter(V[:, rank:])
        
        def hook(module, inputs):
            weight_residual = module.U_residual @ torch.diag(module.S_residual) @ module.V_residual.t()
            module.weight = module.weight_main + weight_residual
        
        layer.register_forward_pre_hook(hook)
        del layer.weight 

    def forward(self, x, return_feature=False, return_energy=False):
        # 【保持修复】强制输入需要梯度，确保 Gradient Checkpointing 生效
        if self.training:
            x.requires_grad_(True)

        features = self.model.vision_model(x)['pooler_output']
        
        if return_feature:
            return features
        
        if self.training:
            return self.fc(features, training=True)
        else:
            logits, e_real, e_fake = self.fc(features, training=False)
            if return_energy:
                return logits, e_real, e_fake
            return logits