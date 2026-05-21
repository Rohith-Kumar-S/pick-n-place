import os
import argparse
import numpy as np
import torch
import sys
from tqdm import tqdm
from pathlib import Path

# Assuming your new Stage 1 modules are saved inside src/
from src.model import SingleViewMAE, patchify, unpatchify, compute_stage1_loss
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

class SingleViewCurriculumDataset(Dataset):
    """
    Pools both Top-Down and Gripper POV images into a single randomized list
    for single-view MAE pre-training. Includes disk caching for fast initialization.
    """
    def __init__(self, data_root: str, img_size: int = 224, is_train: bool = True):
        self.img_size = img_size
        self.is_train = is_train
        
        split_name = "train" if is_train else "test"
        cache_path = os.path.join(data_root, f"stage1_dataset_cache_{split_name}.pt")

        # 1. Check if the cache already exists
        if os.path.exists(cache_path):
            print(f"[{split_name.upper()}] Loading cached dataset from {cache_path}...")
            self.samples = torch.load(cache_path, weights_only=False)
        else:
            # 2. Build the dataset if no cache is found
            self.samples: list[tuple[Path, int, str]] = []
            
            npz_files = sorted(Path(data_root).glob("ep_*.npz"))
            total_npz_files = len(npz_files)
            
            if self.is_train:
                npz_files = npz_files[:int(0.8 * total_npz_files)]
                npz_files = np.random.permutation(npz_files)  # Shuffle episode files
            else:
                npz_files = npz_files[int(0.8 * total_npz_files):]
                
            # Target optimization: adjust debugging ceiling if needed
            print("Istrain: ", self.is_train, " ",len(npz_files))
            # npz_files = npz_files[:200]

            print(f"[{split_name.upper()}] Unpacking {len(npz_files)} NPZ files into a unified single-view pool...")
            
            # Added tqdm here so you don't stare at a blank screen during initial creation
            for npz in tqdm(npz_files, desc=f"Building {split_name} pool"):
                n = np.load(npz, allow_pickle=True)['topdown'].shape[0]
                for i in range(n):
                    # Add BOTH views as separate, independent standalone datapoints
                    self.samples.append((npz, i, 'topdown'))
                    self.samples.append((npz, i, 'gripperpov'))
                    
            # Save the parsed list to disk for the next run
            print(f"[{split_name.upper()}] Saving dataset cache to {cache_path}...")
            torch.save(self.samples, cache_path)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        npz_path, frame_idx, view_key = self.samples[idx]
        ep = np.load(npz_path, allow_pickle=True)
        
        raw_arr = ep[view_key][frame_idx]
        img = Image.fromarray(np.transpose(raw_arr, (1, 2, 0)))
        
        return self.transform(img)

def visualize_stage1_predictions(model, img_tensor, mask_ratio=0.75, num_samples=3, save_path=None):
    """
    Validation Hook: Visualizes the single-view reconstruction pipeline.
    """
    model.eval()
    B = img_tensor.size(0)
    num_samples = min(B, num_samples)
    
    with torch.no_grad():
        preds, mask_indices = model(img_tensor, mask_ratio=mask_ratio)
        
    target_patches = patchify(img_tensor, patch_size=model.patch_size)
    masked_input_patches = target_patches.clone()
    batch_indices = torch.arange(B, device=img_tensor.device).unsqueeze(1).expand(-1, mask_indices.size(1))
    masked_input_patches[batch_indices, mask_indices] = 0.0 
    
    reconstructed_patches = target_patches.clone()
    if preds.size(1) == target_patches.size(1):
        reconstructed_patches[batch_indices, mask_indices] = preds[batch_indices, mask_indices]
    else:
        reconstructed_patches[batch_indices, mask_indices] = preds
        
    masked_input_img = unpatchify(masked_input_patches, patch_size=model.patch_size)
    reconstructed_img = unpatchify(reconstructed_patches, patch_size=model.patch_size)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
    
    def unnorm(img):
        img = img * std + mean
        return torch.clamp(img, 0, 1).cpu().numpy()
        
    orig_vis = unnorm(img_tensor)
    masked_vis = unnorm(masked_input_img)
    recon_vis = unnorm(reconstructed_img)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3.5 * num_samples))
    if num_samples == 1: axes = [axes]
        
    for i in range(num_samples):
        axes[i][0].imshow(np.transpose(masked_vis[i], (1, 2, 0)))
        axes[i][0].set_title(f"Masked Input ({int(mask_ratio*100)}%)" if i==0 else "")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(np.transpose(recon_vis[i], (1, 2, 0)))
        axes[i][1].set_title("Reconstruction" if i==0 else "")
        axes[i][1].axis('off')
        
        axes[i][2].imshow(np.transpose(orig_vis[i], (1, 2, 0)))
        axes[i][2].set_title("Ground Truth" if i==0 else "")
        axes[i][2].axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser("Stage 1 Single-View Curriculum MAE Pretraining")
    parser.add_argument("--expt", type=str, default="expt_4_stage1", help="expt name")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=150, help="Stage 1 training total epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Starting learning rate")
    return parser.parse_args()


def main():
    arglist = parse_args()

    np.random.seed(arglist.seed)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(arglist.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define dedicated directory structures to avoid collision with Stage 2 outputs
    model_dir = os.path.join("/content/drive/MyDrive/APLDL/models/", arglist.expt)
    results_dir = os.path.join("/content/drive/MyDrive/APLDL/results/", arglist.expt)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Initialize the SingleView Curriculum Model
    model = SingleViewMAE(patch_size=8, embed_dim=512, num_heads=4, enc_depth=6, dec_depth=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arglist.lr, weight_decay=0.05)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Initialize the flattened single-view datasets
    train_data = SingleViewCurriculumDataset("/content/drive/MyDrive/APLDL/new_data_1/raw/expt_4/", is_train=True)
    test_data = SingleViewCurriculumDataset("/content/drive/MyDrive/APLDL/new_data_1/raw/expt_4/", is_train=False)

    num_workers = 2 if torch.cuda.is_available() else 0
    pin_memory = True if torch.cuda.is_available() else False

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=arglist.batch_size, shuffle=True, 
                                               num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=arglist.batch_size, shuffle=True, 
                                              num_workers=num_workers, pin_memory=pin_memory)
    print("Data loading pool fully built.")

    best_test_loss = np.inf
    
    for epoch in range(arglist.epochs):
        print(f"Epoch {epoch + 1} / {arglist.epochs}")
        
        # ==========================================
        # TRAINING LOOP
        # ==========================================
        model.train()
        train_loss = []
        for img_batch in tqdm(train_loader, total=len(train_loader), desc="Stage 1 Training"):
            img_batch = img_batch.to(device)
            optimizer.zero_grad()

            # Execute single view loss sequence at 75% standard masking ratio
            loss = compute_stage1_loss(model, img_batch, mask_ratio=0.4)
            loss.backward()

            # Enforce hard-stop value clipping to ground multimodal dynamics safely
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

            train_loss.append(loss.item())
    
        train_loss = np.array(train_loss).mean()

        # ==========================================
        # TESTING/VALIDATION LOOP
        # ==========================================
        model.eval()
        with torch.no_grad():
            test_loss = []
            for img_batch in tqdm(test_loader, total=len(test_loader), desc="Stage 1 Testing"):
                img_batch = img_batch.to(device)
                loss = compute_stage1_loss(model, img_batch, mask_ratio=0.4)
                test_loss.append(loss.item())
                
        test_loss = np.array(test_loss).mean()
        print(f"train loss: {train_loss:.6f} | test loss: {test_loss:.6f}")
        
        scheduler.step(test_loss)
        
        # Save checkpoints safely based on optimization plateaus
        if test_loss < best_test_loss:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'epoch': epoch
            }, os.path.join(model_dir, "stage1_best.ckpt"))
            best_test_loss = test_loss
        
        # Step checkpoint saving and visual inspection dump
        if epoch % 5 == 0 or epoch == arglist.epochs - 1:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'epoch': epoch
            }, os.path.join(model_dir, f"stage1_epoch_{epoch}.ckpt"))
            
            print("Dumping Visual Validation Reconstructions...")
            visualize_stage1_predictions(
                model, img_batch, mask_ratio=0.4, num_samples=3, 
                save_path=os.path.join(results_dir, f"stage1_epoch_{epoch}.png")
            )

if __name__ == '__main__':
    main()