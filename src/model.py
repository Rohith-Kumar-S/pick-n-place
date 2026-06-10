""""
Rohith Kumar Senthil Kumar

Pick and Place with CroCo-Enhanced Vector Field Learning Models
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CroCoAutoencoder(nn.Module):
    def __init__(self, img_size=224, patch_size=8, embed_dim=512, num_heads=4, enc_depth=6, dec_depth=6):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Hierarchical extraction prevents blocks from blurring into the background
        hidden_dim = embed_dim // 2
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=4, stride=4),
            nn.LayerNorm([hidden_dim, img_size // 4, img_size // 4]), 
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=2, stride=2)
        )
        
        self.patch_norm = nn.LayerNorm(embed_dim)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.grip_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.grip_pos_embed, std=.02)

        self.topdown_view_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.gripper_view_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        torch.nn.init.trunc_normal_(self.topdown_view_embed, std=.02)
        torch.nn.init.trunc_normal_(self.gripper_view_embed, std=.02)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.trunc_normal_(self.mask_token, std=.02)
        
        # Transformer block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, norm_first=True  
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_depth)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, norm_first=True  
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_depth)
        
        self.reconstruction_head = nn.Linear(embed_dim, 3 * patch_size * patch_size)

    def forward(self, topdown_img, gripper_img, top_mask_ratio=0.95, grip_mask_ratio=0.40, return_embedding_only=False):
        B = topdown_img.size(0)
        
        #  RAW FEATURE EXTRACTION 
        top_tokens = self.patch_embed(topdown_img).flatten(2).transpose(1, 2)
        grip_tokens = self.patch_embed(gripper_img).flatten(2).transpose(1, 2)

        top_tokens = self.patch_norm(top_tokens)
        grip_tokens = self.patch_norm(grip_tokens)

        # ENCODER input assembly (with view-specific embeddings)    
        enc_top_tokens = top_tokens + self.pos_embed + self.topdown_view_embed
        enc_grip_tokens = grip_tokens + self.grip_pos_embed + self.gripper_view_embed
        
        # TOP-DOWN MASKING (95%)
        num_masked_top = int(top_mask_ratio * self.num_patches)
        noise_top = torch.rand(B, self.num_patches, device=top_tokens.device)
        mask_indices_top = torch.argsort(noise_top, dim=1)[:, :num_masked_top]
        visible_indices_top = torch.argsort(noise_top, dim=1)[:, num_masked_top:]
        
        batch_indices_top = torch.arange(B).unsqueeze(1).expand(-1, self.num_patches - num_masked_top)
        visible_top_tokens = enc_top_tokens[batch_indices_top, visible_indices_top]

        # GRIPPER MASKING (40%)
        num_masked_grip = int(grip_mask_ratio * self.num_patches)
        noise_grip = torch.rand(B, self.num_patches, device=grip_tokens.device)
        # We only need the visible indices to pass to the encoder
        visible_indices_grip = torch.argsort(noise_grip, dim=1)[:, num_masked_grip:]
        
        batch_indices_grip = torch.arange(B).unsqueeze(1).expand(-1, self.num_patches - num_masked_grip)
        visible_grip_tokens = enc_grip_tokens[batch_indices_grip, visible_indices_grip]
        
        # 4. ENCODING
        # Encoder now receives 5% of top-down tokens, and 60% of gripper tokens
        enc_top = self.encoder(visible_top_tokens)
        enc_grip = self.encoder(visible_grip_tokens)
        
        # DECODER input assembly: we inject the mask token back into the sequence to maintain positional alignment
        full_top_tokens = self.mask_token.expand(B, self.num_patches, -1).clone()
        full_top_tokens[batch_indices_top, visible_indices_top] = enc_top
        full_top_tokens = full_top_tokens + self.pos_embed + self.topdown_view_embed
        
        # CROSS-VIEW COMPLETION ---
        fused_embeddings = self.decoder(tgt=full_top_tokens, memory=enc_grip)
        
        if return_embedding_only:
            return fused_embeddings.mean(dim=1) 
            
        # RECONSTRUCTION 
        reconstructed_patches = self.reconstruction_head(fused_embeddings)
        
        # We return the top-down mask indices so the loss function knows which patches to penalize
        return reconstructed_patches, mask_indices_top
    
def patchify(imgs, patch_size=16):
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x) 
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3)) 
    return x

def unpatchify(x, patch_size=16):
    B = x.shape[0]
    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    x = x.reshape(shape=(B, h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    x = x.reshape(shape=(B, 3, h * p, w * p))
    return x
def compute_croco_loss(model, topdown_img, gripper_img, bbox, top_mask_ratio=0.95, grip_mask_ratio=0.40):
    """
    Color-Heuristic Boosted L1 Loss. 
    Massively penalizes the network for failing to reconstruct multi-channel neon blocks.
    """
    B = topdown_img.size(0)
    P = model.patch_size
    num_patches_1d = topdown_img.size(2) // P
    
    # Forward Pass
    preds, mask_indices = model(topdown_img, gripper_img, top_mask_ratio=top_mask_ratio, grip_mask_ratio=grip_mask_ratio)
    target_patches = model.patchify(topdown_img) if hasattr(model, 'patchify') else patchify(topdown_img, patch_size=P)
    batch_indices = torch.arange(B, device=topdown_img.device).unsqueeze(1).expand(-1, mask_indices.size(1))
    
    masked_targets = target_patches[batch_indices, mask_indices]
    
    if preds.size(1) == target_patches.size(1):
        masked_preds = preds[batch_indices, mask_indices] 
    else:
        masked_preds = preds 
        
    # Base L1 Error
    raw_patch_loss = torch.abs(masked_preds - masked_targets).mean(dim=-1) # [B, num_masked]
    
    # Create a base weight matrix of 1.0 for the 2D patch grid: shape [B, grid_h, grid_w]
    spatial_weights = torch.ones((B, num_patches_1d, num_patches_1d), device=topdown_img.device)
    
    # Iterate over the batch to apply the 10x penalty to RoIs
    for b in range(B):
        for box in bbox[b]:
            x1, y1, x2, y2 = box.int()
            
            # Skip padded/empty boxes
            if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
                continue
            
            # Convert pixel coords to patch grid indices (safely clamped to grid bounds)
            px1 = torch.clamp(x1 // P, 0, num_patches_1d - 1)
            px2 = torch.clamp(x2 // P, 0, num_patches_1d - 1)
            py1 = torch.clamp(y1 // P, 0, num_patches_1d - 1)
            py2 = torch.clamp(y2 // P, 0, num_patches_1d - 1)
            
            # Apply 10x penalty to the spatial regions containing the objects
            spatial_weights[b, py1:py2+1, px1:px2+1] = 50.0
            
    # Flatten spatial weights to match the 1D patch sequence: [B, total_patches]
    flat_weights = spatial_weights.view(B, -1)
    
    # Gather ONLY the weights for the patches that were actually masked out
    masked_weights = flat_weights[batch_indices, mask_indices] # Shape: [B, num_masked]
    
    # Apply the exact spatial boost
    boosted_loss = raw_patch_loss * masked_weights
    
    return boosted_loss.mean()

def visualize_croco_predictions(model, topdown_img, gripper_img, top_mask_ratio=0.95, grip_mask_ratio=0.40, num_samples=3, save_path=None):
    model.eval()
    B = topdown_img.size(0)
    num_samples = min(B, num_samples) 
    
    with torch.no_grad():
        preds, mask_indices = model(topdown_img, gripper_img, top_mask_ratio=top_mask_ratio, grip_mask_ratio=grip_mask_ratio)
        
    target_patches = patchify(topdown_img, patch_size=model.patch_size)
    masked_input_patches = target_patches.clone()
    batch_indices = torch.arange(B, device=topdown_img.device).unsqueeze(1).expand(-1, mask_indices.size(1))
    masked_input_patches[batch_indices, mask_indices] = 0.0 
    
    reconstructed_patches = target_patches.clone()
    
    # Fix alignment for visualizer just like in the loss function
    if preds.size(1) == target_patches.size(1):
        reconstructed_patches[batch_indices, mask_indices] = preds[batch_indices, mask_indices]
    else:
        reconstructed_patches[batch_indices, mask_indices] = preds
    
    masked_input_img = unpatchify(masked_input_patches, patch_size=model.patch_size)
    reconstructed_img = unpatchify(reconstructed_patches, patch_size=model.patch_size)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=topdown_img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=topdown_img.device).view(1, 3, 1, 1)
    
    def unnorm(img):
        img = img * std + mean
        return torch.clamp(img, 0, 1).cpu().numpy() 
        
    topdown_vis = unnorm(topdown_img)
    gripper_vis = unnorm(gripper_img)
    masked_vis = unnorm(masked_input_img)
    recon_vis = unnorm(reconstructed_img)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.5 * num_samples))
    if num_samples == 1: axes = [axes] 
        
    for i in range(num_samples):
        axes[i][0].imshow(np.transpose(gripper_vis[i], (1, 2, 0)))
        axes[i][0].set_title(f"Gripper Context ({int(grip_mask_ratio*100)}% Masked Internally)" if i==0 else "")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(np.transpose(masked_vis[i], (1, 2, 0)))
        # FIX: Changed mask_ratio to top_mask_ratio here
        axes[i][1].set_title(f"Masked Top-Down ({int(top_mask_ratio*100)}%)" if i==0 else "")
        axes[i][1].axis('off')
        
        axes[i][2].imshow(np.transpose(recon_vis[i], (1, 2, 0)))
        axes[i][2].set_title("Reconstruction" if i==0 else "")
        axes[i][2].axis('off')
        
        axes[i][3].imshow(np.transpose(topdown_vis[i], (1, 2, 0)))
        axes[i][3].set_title("Ground Truth" if i==0 else "")
        axes[i][3].axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)
    
    
class SingleViewMAE(nn.Module):
    def __init__(self, img_size=224, patch_size=8, embed_dim=512, num_heads=4, enc_depth=6, dec_depth=6):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Exact same layers as CroCo Stage 2
        hidden_dim = embed_dim // 2
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=4, stride=4),
            nn.LayerNorm([hidden_dim, img_size // 4, img_size // 4]), 
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=2, stride=2)
        )
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        
        # Shared Encoder 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, norm_first=True  
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_depth)
        
        # STAGE 1 SPECIFIC DECODER (Self-Attention)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.trunc_normal_(self.mask_token, std=.02)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=dec_depth)
        self.reconstruction_head = nn.Linear(embed_dim, 3 * patch_size * patch_size)

    def forward(self, img, mask_ratio=0.75):
        B = img.size(0)
        
        # Extract and normalize patches
        tokens = self.patch_embed(img).flatten(2).transpose(1, 2)
        tokens = self.patch_norm(tokens)
        
        # Apply spatial mapping
        tokens = tokens + self.pos_embed
        
        # Masking sequence
        num_masked = int(mask_ratio * self.num_patches)
        noise = torch.rand(B, self.num_patches, device=img.device)
        mask_indices = torch.argsort(noise, dim=1)[:, :num_masked]
        visible_indices = torch.argsort(noise, dim=1)[:, num_masked:]
        
        batch_indices = torch.arange(B).unsqueeze(1).expand(-1, self.num_patches - num_masked)
        visible_tokens = tokens[batch_indices, visible_indices]
        
        # Encode unmasked patches
        encoded_tokens = self.encoder(visible_tokens)
        
        # Reassemble the sequence for the decoder
        full_tokens = self.mask_token.expand(B, self.num_patches, -1).clone()
        full_tokens[batch_indices, visible_indices] = encoded_tokens
        
        # Reinject coordinates strictly once before reconstruction
        full_tokens = full_tokens + self.pos_embed
        
        # Decode and process reconstruction
        decoded_tokens = self.decoder(full_tokens)
        preds = self.reconstruction_head(decoded_tokens)
        
        return preds, mask_indices
    
def compute_stage1_loss(model, img, bbox, mask_ratio=0.75):
    """
    RoI-Boosted Stage 1 Loss.
    Forces the Single-View MAE to prioritize the exact bounding box regions.
    """
    B = img.size(0)
    P = model.patch_size
    num_patches_1d = img.size(2) // P  # Assuming square image: H = W
    
    # Single-image forward pass
    preds, mask_indices = model(img, mask_ratio=mask_ratio)
    target_patches = patchify(img, patch_size=P)
    
    batch_indices = torch.arange(B, device=img.device).unsqueeze(1).expand(-1, mask_indices.size(1))
    masked_targets = target_patches[batch_indices, mask_indices]
    
    if preds.size(1) == target_patches.size(1):
        masked_preds = preds[batch_indices, mask_indices]
    else:
        masked_preds = preds
        
    # Base L1 Error per patch
    # Shape: [B, num_masked]
    raw_patch_loss = torch.abs(masked_preds - masked_targets).mean(dim=-1) 
    
    # EXACT ROI BOUNDING BOX BOOST 
    # Create a base weight matrix of 1.0 for the 2D patch grid: shape [B, grid_h, grid_w]
    spatial_weights = torch.ones((B, num_patches_1d, num_patches_1d), device=img.device)
    
    # Iterate over the batch to apply the 10x penalty to RoIs
    for b in range(B):
        for box in bbox[b]:
            x1, y1, x2, y2 = box.int()
            
            # Skip padded/empty boxes
            if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
                continue
            
            # Convert pixel coords to patch grid indices (safely clamped to grid bounds)
            px1 = torch.clamp(x1 // P, 0, num_patches_1d - 1)
            px2 = torch.clamp(x2 // P, 0, num_patches_1d - 1)
            py1 = torch.clamp(y1 // P, 0, num_patches_1d - 1)
            py2 = torch.clamp(y2 // P, 0, num_patches_1d - 1)
            
            # Apply 10x penalty to the spatial regions containing the objects
            spatial_weights[b, py1:py2+1, px1:px2+1] = 10.0
            
    # Flatten spatial weights to match the 1D patch sequence: [B, total_patches]
    flat_weights = spatial_weights.view(B, -1)
    
    # Gather ONLY the weights for the patches that were actually masked out
    masked_weights = flat_weights[batch_indices, mask_indices] # Shape: [B, num_masked]
    
    # Apply the exact spatial boost
    boosted_loss = raw_patch_loss * masked_weights
    
    return boosted_loss.mean()

class Stage3VectorField(nn.Module):
    def __init__(self, arglist):
        super().__init__()
        self.arglist = arglist 
        
        # Observation Encoders
        # We project all distinct inputs into a common hidden dimension (d_emb)
        self.proprio_encoder = nn.Linear(arglist.d_proprio, arglist.d_emb)
        
        if arglist.image:
            # Assuming your CroCo embedding is 128D (change to 512 or 768 if different)
            self.croco_encoder = nn.Linear(512, arglist.d_emb)
            
        if arglist.text:
            # Simple, efficient categorical lookup table for the target (0, 1, or 2)
            self.text_encoder = nn.Embedding(arglist.num_objects, arglist.d_emb)

        # Flow Matching Encoders
        self.time_encoder = nn.Linear(1, arglist.d_emb)
        self.action_encoder = nn.Linear(arglist.d_act, arglist.d_emb)

        # Calculate total input dimension dynamically
        num_inputs = 2 # proprio + time + action (wait, action is part of the state in FM)
        num_inputs = 1 # proprio
        if arglist.image: num_inputs += 1
        if arglist.text: num_inputs += 1
        num_inputs += 2 # action + time
        
        input_dim = arglist.d_emb * num_inputs

        # Core Vector Field MLP
        layers = [nn.Linear(input_dim, arglist.d_model), nn.SiLU()]
        for _ in range(arglist.num_layers - 2):
            layers += [nn.Linear(arglist.d_model, arglist.d_model), nn.SiLU()]
        layers.append(nn.Linear(arglist.d_model, arglist.d_act))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, O, A, tau):
        """
        O['proprio']: [B, 4]
        O['croco_embedding']: [B, 128]
        O['text']: [B] (Target ID: 0, 1, or 2)
        A: [B, 4] (The noisy action)
        tau: [B, 1] (Time scalar)
        """
        obs_emb = []
        
        # Encode Proprioception
        obs_emb.append(self.proprio_encoder(O['proprio']))

        # Encode Visuals (No CNNs needed, just a linear projection!)
        if self.arglist.image:
            obs_emb.append(self.croco_encoder(O['croco_embedding']))
        
        # Encode Language/Target
        if self.arglist.text:
            # Ensure text is a 1D long tensor for the Embedding layer
            text_idx = O['text'].long()
            if text_idx.dim() > 1:
                text_idx = text_idx.squeeze(-1)
            obs_emb.append(self.text_encoder(text_idx))
        
        # Encode Action and Time
        obs_emb.append(self.action_encoder(A))
        obs_emb.append(self.time_encoder(tau))

        # Concatenate and pass through MLP
        fused_state = torch.cat(obs_emb, dim=-1)
        v = self.mlp(fused_state)
        
        return v


class FlowMatchingVLA(nn.Module):
    def __init__(self, arglist):
        super().__init__()
        self.arglist = arglist
        self.vector_field = Stage3VectorField(arglist)
        
        # Load pre-computed statistics for normalization
        data_dir = os.path.join("/content/drive/MyDrive/APLDL/data/raw/train")
        self.stats = np.load(os.path.join(data_dir, "stats.npz"), allow_pickle=True)
    
    def loss(self, O, A):
        # Sample noise and time
        eps = torch.randn_like(A)
        tau = torch.rand_like(A[:, :1])
        
        # Create the noisy action trajectory
        A_noisy = tau * A + (1 - tau) * eps
        
        # Predict the flow vector
        action_preds = self.vector_field(O, A_noisy, tau)
        
        # Standard Optimal Transport (OT) Flow Matching target
        action_target = A - eps
        
        # Clean, single-objective loss
        action_loss = nn.functional.mse_loss(action_preds, action_target)
        
        return action_loss

    def rk1(self, O, A, tau, h): # Euler Method
        k1 = self.vector_field(O, A, tau)
        return A + h * k1

    def rk2(self, O, A, tau, h): # Ralston's method
        k1 = self.vector_field(O, A, tau)
        k2 = self.vector_field(O, A + h * k1, tau + h)
        alpha = 2.0 / 3.0 
        return A + h * ((1.0 - 1.0/(2.0*alpha))*k1 + (1.0/(2.0*alpha))*k2)

    @torch.no_grad()
    def sample(self, O_dict, device):
        """
        O_dict should now just contain the extracted 1D tensors:
        O_dict = {'proprio': [...], 'croco_embedding': [...], 'text': [...]}
        """
        n_samples = 1 # Assuming batch size 1 for live inference
        h = 1 / self.arglist.T_flow
        tau = torch.zeros(n_samples, 1, device=device)
        A = torch.randn(n_samples, self.arglist.d_act, device=device)
        
        # Simulate the ODE forward in time
        with torch.no_grad():
            for _ in range(self.arglist.T_flow):
                A = self.rk2(O_dict, A, tau, h)
                tau = tau + h
                
        # Denormalize the predicted action using train statistics
        a = A.cpu().numpy()[0]
        if self.arglist.normalize:
            a = a * self.stats['action_std'] + self.stats['action_mean']
            
        return a