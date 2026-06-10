"""
Rohith Kumar Senthil Kumar
VLA Training Script
Run a specific stage with: python train.py --stage {1,2,3}
"""

import os
import argparse
import numpy as np
import torch
import sys
from tqdm import tqdm
from pathlib import Path

import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from src.model import (
    SingleViewMAE, patchify, unpatchify, compute_stage1_loss,
    CroCoAutoencoder, compute_croco_loss, visualize_croco_predictions,
    FlowMatchingVLA,
)
from src.utils import normalize, get_tensor, create_patch_mask, apply_patch_mask


# STAGE 1 — Dataset
class SingleViewCurriculumDataset(Dataset):
    """
    Pools both Top-Down and Gripper POV images into a single randomised list
    for single-view MAE pre-training. Includes disk caching for fast init.
    """

    def __init__(self, data_root: str, img_size: int = 224, is_train: bool = True):
        self.img_size = img_size
        self.is_train = is_train

        split_name = "train" if is_train else "test"
        cache_path = os.path.join(data_root, f"stage1_dataset_cache_{split_name}.pt")

        if os.path.exists(cache_path):
            print(f"[{split_name.upper()}] Loading cached dataset from {cache_path}...")
            self.samples = torch.load(cache_path, weights_only=False)
        else:
            self.samples: list[tuple[Path, int, str]] = []
            npz_files = sorted(Path(data_root).glob("ep_*.npz"))
            total_npz_files = len(npz_files)

            if self.is_train:
                npz_files = npz_files[:int(0.8 * total_npz_files)]
                npz_files = np.random.permutation(npz_files)
            else:
                npz_files = npz_files[int(0.8 * total_npz_files):]

            print("is_train:", self.is_train, " | files:", len(npz_files))
            print(f"[{split_name.upper()}] Unpacking {len(npz_files)} NPZ files...")

            for npz in tqdm(npz_files, desc=f"Building {split_name} pool"):
                n = np.load(npz, allow_pickle=True)['topdown'].shape[0]
                for i in range(n):
                    self.samples.append((npz, i, 'topdown'))
                    self.samples.append((npz, i, 'gripperpov'))

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
        img_tensor = self.transform(img)

        bbox_key = 'bboxes_top' if view_key == 'topdown' else 'bboxes_gripper'
        bboxes = ep[bbox_key][frame_idx]
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)

        return img_tensor, bboxes_tensor



# STAGE 2 — Dataset
class CroCoPairDataset(Dataset):
    """
    Loads (topdown_full, gripperpov_full, bboxes) triplets for CroCo-style
    cross-view pretraining. Masking is applied on-the-fly inside the loss fn.
    """

    def __init__(
        self,
        data_root: str,
        patch_size: int = 8,
        mask_ratio: float = 0.90,
        img_size: int = 224,
        use_saved_masks: bool = False,
        is_train: bool = True,
    ):
        assert img_size % patch_size == 0, \
            f"img_size {img_size} must be divisible by patch_size {patch_size}"

        self.patch_size      = patch_size
        self.mask_ratio      = mask_ratio
        self.img_size        = img_size
        self.use_saved_masks = use_saved_masks
        self.is_train        = is_train

        self.samples: list[tuple[Path, int]] = []
        npz_files = sorted(Path(data_root).glob("Copy of ep_*.npz"))
        print(f"Loading {len(npz_files)} NPZ files...")

        for npz in npz_files:
            n = np.load(npz, allow_pickle=True)['topdown'].shape[0]
            for i in range(n):
                self.samples.append((npz, i))

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        npz_path, frame_idx = self.samples[idx]
        ep = np.load(npz_path, allow_pickle=True)

        topdown_full = Image.fromarray(np.transpose(ep['topdown'][frame_idx], (1, 2, 0)))
        gripper_full = Image.fromarray(np.transpose(ep['gripperpov'][frame_idx], (1, 2, 0)))

        topdown_full_t = self.transform(topdown_full)
        gripper_full_t = self.transform(gripper_full)

        bboxes = ep['bboxes_top'][frame_idx]
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)

        return topdown_full_t, gripper_full_t, bboxes_tensor


# STAGE 3 — Dataset
class Stage3Dataset(Dataset):
    """
    Loads pre-extracted CroCo embeddings + proprio + actions for Stage 3
    flow-matching policy training.
    """

    def __init__(self, arglist, mode: str):
        super().__init__()
        self.arglist = arglist

        data_dir = os.path.join("/content/drive/MyDrive/APLDL/data/", "raw")
        data_path = os.path.join(data_dir, mode, f"{mode}.npz")
        stats_path = os.path.join(data_dir, "train", "stats.npz")

        print(f"Loading {mode} data from: {data_path}")
        dataset = np.load(data_path, allow_pickle=True)
        stats   = np.load(stats_path, allow_pickle=True)

        self.proprio = dataset['proprio']
        self.A       = dataset['action']

        if self.arglist.image:
            self.croco_embedding = dataset['croco_embedding']
        if self.arglist.text:
            self.text = dataset['text']

        if self.arglist.normalize:
            self.proprio_mean = stats['proprio_mean']
            self.proprio_std  = stats['proprio_std']
            self.A_mean       = stats['action_mean']
            self.A_std        = stats['action_std']

        self.dims = self.A.shape

    def __len__(self):
        return self.dims[0]

    def __getitem__(self, idx):
        if self.arglist.normalize:
            o = {'proprio': get_tensor(normalize(self.proprio[idx], self.proprio_mean, self.proprio_std))}
            a = get_tensor(normalize(self.A[idx], self.A_mean, self.A_std))
        else:
            o = {'proprio': get_tensor(self.proprio[idx])}
            a = get_tensor(self.A[idx])

        if self.arglist.image:
            o['croco_embedding'] = get_tensor(self.croco_embedding[idx])
        if self.arglist.text:
            o['text'] = torch.tensor(self.text[idx], dtype=torch.long)

        return o, a


def stage3_collate_fn(batch):
    Os, As = zip(*batch)
    O_out = {k: torch.stack([o[k] for o in Os]) for k in Os[0].keys()}
    return O_out, torch.stack(As)


# STAGE 1 — Visualisation helper
def visualize_stage1_predictions(model, img_tensor, mask_ratio=0.75, num_samples=3, save_path=None):
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

    masked_input_img  = unpatchify(masked_input_patches,  patch_size=model.patch_size)
    reconstructed_img = unpatchify(reconstructed_patches, patch_size=model.patch_size)

    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)

    def unnorm(img):
        return torch.clamp(img * std + mean, 0, 1).cpu().numpy()

    orig_vis   = unnorm(img_tensor)
    masked_vis = unnorm(masked_input_img)
    recon_vis  = unnorm(reconstructed_img)

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3.5 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        axes[i][0].imshow(np.transpose(masked_vis[i], (1, 2, 0)))
        axes[i][0].set_title(f"Masked Input ({int(mask_ratio*100)}%)" if i == 0 else "")
        axes[i][0].axis('off')

        axes[i][1].imshow(np.transpose(recon_vis[i], (1, 2, 0)))
        axes[i][1].set_title("Reconstruction" if i == 0 else "")
        axes[i][1].axis('off')

        axes[i][2].imshow(np.transpose(orig_vis[i], (1, 2, 0)))
        axes[i][2].set_title("Ground Truth" if i == 0 else "")
        axes[i][2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)



# ARG PARSING
def parse_args():
    parser = argparse.ArgumentParser("Unified VLA Training — Stages 1 / 2 / 3")
    parser.add_argument("--stage",      type=int,   required=True, choices=[1, 2, 3], help="Training stage to run")
    parser.add_argument("--expt",       type=str,   default="expt_1",   help="Experiment name (determines checkpoint dir)")
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--ckpt",       type=str,   default="",         help="Checkpoint filename to resume from")

    # Stage 3 specific arguments 
    parser.add_argument("--env",          type=str,   default="bin-picking-three-objects-v3")
    parser.add_argument("--d-proprio",    type=int,   default=4)
    parser.add_argument("--d-act",        type=int,   default=4)
    parser.add_argument("--image",        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text",         action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--T-flow",       type=int,   default=20)
    parser.add_argument("--d-model",      type=int,   default=256)
    parser.add_argument("--d-emb",        type=int,   default=256)
    parser.add_argument("--normalize",    action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_layers",   type=int,   default=4)
    parser.add_argument("--num_objects",  type=int,   default=3)

    return parser.parse_args()


# STAGE RUNNERS
def run_stage1(arglist, device, model_dir, results_dir):
    model = SingleViewMAE(
        patch_size=8, embed_dim=512, num_heads=4, enc_depth=6, dec_depth=6
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arglist.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    start_epoch = 0

    if arglist.ckpt:
        ckpt_path = os.path.join(model_dir, arglist.ckpt)
        print(f"Loading Stage 1 checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    data_root = "/content/drive/MyDrive/APLDL/new_data_2/raw/expt_4/"
    train_data = SingleViewCurriculumDataset(data_root, is_train=True)
    test_data  = SingleViewCurriculumDataset(data_root, is_train=False)

    num_workers = 2 if torch.cuda.is_available() else 0
    pin_memory  = torch.cuda.is_available()
    train_loader = DataLoader(train_data, batch_size=arglist.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_data,  batch_size=arglist.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    print("Stage 1 data loaded.")

    best_test_loss = np.inf

    for epoch in range(start_epoch, arglist.epochs):
        print(f"Epoch {epoch + 1} / {arglist.epochs}")

        # Train
        model.train()
        train_losses = []
        for img_batch, bbox_batch in tqdm(train_loader, desc="Stage 1 Training"):
            img_batch  = img_batch.to(device)
            bbox_batch = bbox_batch.to(device)
            optimizer.zero_grad()
            loss = compute_stage1_loss(model, img_batch, bbox_batch, mask_ratio=0.4)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        # Eval 
        model.eval()
        test_losses = []
        with torch.no_grad():
            for img_batch, bbox_batch in tqdm(test_loader, desc="Stage 1 Testing"):
                img_batch  = img_batch.to(device)
                bbox_batch = bbox_batch.to(device)
                loss = compute_stage1_loss(model, img_batch, bbox_batch, mask_ratio=0.4)
                test_losses.append(loss.item())
        test_loss = np.mean(test_losses)

        print(f"train loss: {train_loss:.6f} | test loss: {test_loss:.6f}")
        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                       os.path.join(model_dir, "stage1_best.ckpt"))
            best_test_loss = test_loss

        if epoch % 5 == 0 or epoch == arglist.epochs - 1:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                       os.path.join(model_dir, f"stage1_epoch_{epoch}.ckpt"))
            print("Dumping visual validation reconstructions...")
            visualize_stage1_predictions(
                model, img_batch, mask_ratio=0.4, num_samples=3,
                save_path=os.path.join(results_dir, f"stage1_epoch_{epoch}.png")
            )


def run_stage2(arglist, device, model_dir, results_dir):
    model = CroCoAutoencoder().to(device)
    checkpoint_path = os.path.join(model_dir, arglist.ckpt) if arglist.ckpt else ""

    if checkpoint_path and os.path.exists(checkpoint_path):
        #  Resume Stage 2 
        print(f"\n[RESUME] Found Stage 2 checkpoint at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        for name, param in model.named_parameters():
            if 'patch_embed' in name or 'encoder' in name:
                param.requires_grad = False
        print("[RESUME] Encoder frozen.")
        optimizer   = torch.optim.AdamW(model.parameters(), lr=arglist.lr, weight_decay=0.05)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[RESUME] Resuming from epoch {start_epoch}")

    else:
        # Curriculum handoff from Stage 1
        print(f"\n[START] No Stage 2 checkpoint found. Attempting curriculum handoff from Stage 1...")
        stage1_ckpt_path = f"/content/drive/MyDrive/APLDL/models/{arglist.expt}/stage1_best.ckpt"

        if os.path.exists(stage1_ckpt_path):
            checkpoint    = torch.load(stage1_ckpt_path, map_location=device)
            stage1_weights = checkpoint['model']
            croco_state    = model.state_dict()
            transfer_dict  = {
                k: v for k, v in stage1_weights.items()
                if 'decoder' not in k and k in croco_state and croco_state[k].shape == v.shape
            }
            croco_state.update(transfer_dict)
            model.load_state_dict(croco_state)
            print(f"[CURRICULUM] Transferred {len(transfer_dict)} parameter tensors from Stage 1.")
        else:
            print("[WARNING] Stage 1 checkpoint not found. Initialising CroCo from scratch.")

        for name, param in model.named_parameters():
            if 'patch_embed' in name or 'encoder' in name:
                param.requires_grad = False
        print("[CURRICULUM] Encoder frozen. Training decoder only.")
        optimizer   = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=arglist.lr, weight_decay=0.05)
        start_epoch = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    train_data = CroCoPairDataset("/content/drive/MyDrive/APLDL/new_data_2/raw/expt_5/", is_train=True)
    test_data  = CroCoPairDataset("/content/drive/MyDrive/APLDL/new_data_2/raw/test/",   is_train=False)

    num_workers = 2 if torch.cuda.is_available() else 0
    pin_memory  = torch.cuda.is_available()
    train_loader = DataLoader(train_data, batch_size=arglist.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_data,  batch_size=arglist.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    print("Stage 2 data loaded.")

    best_test_loss  = np.inf
    best_train_loss = np.inf
    early_stop_patience    = 4
    train_patience_counter = 0

    for epoch in range(start_epoch, arglist.epochs):
        print(f"Epoch {epoch + 1} / {arglist.epochs}")

        # Train
        model.train()
        train_losses = []
        for full_topdown, gripper_pov, bboxes_tensor in tqdm(train_loader, desc="Training"):
            full_topdown   = full_topdown.to(device)
            gripper_pov    = gripper_pov.to(device)
            bboxes_tensor  = bboxes_tensor.to(device)
            optimizer.zero_grad()
            loss = compute_croco_loss(model, full_topdown, gripper_pov, bboxes_tensor,
                                      top_mask_ratio=0.95, grip_mask_ratio=0.40)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        if train_loss < best_train_loss:
            best_train_loss        = train_loss
            train_patience_counter = 0
        else:
            train_patience_counter += 1
            print(f"-> Train loss plateau. Patience: {train_patience_counter}/{early_stop_patience}")

        # Eval
        model.eval()
        test_losses = []
        with torch.no_grad():
            for full_topdown, gripper_pov, bboxes_tensor in tqdm(test_loader, desc="Testing"):
                full_topdown  = full_topdown.to(device)
                gripper_pov   = gripper_pov.to(device)
                bboxes_tensor = bboxes_tensor.to(device)
                loss = compute_croco_loss(model, full_topdown, gripper_pov, bboxes_tensor,
                                         top_mask_ratio=0.95, grip_mask_ratio=0.40)
                test_losses.append(loss.item())
        test_loss = np.mean(test_losses)

        print(f"train loss: {train_loss:.4f} | test loss: {test_loss:.4f}")
        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                       os.path.join(model_dir, "best.ckpt"))
            best_test_loss = test_loss

        if epoch % 5 == 0 or epoch == arglist.epochs - 1:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                       os.path.join(model_dir, f"{epoch}.ckpt"))
            print("Visualising reconstructions...")
            visualize_croco_predictions(
                model, full_topdown, gripper_pov,
                top_mask_ratio=0.95, grip_mask_ratio=0.40, num_samples=3,
                save_path=os.path.join(results_dir, f"stage2_epoch_{epoch}.png")
            )

        if train_patience_counter >= early_stop_patience:
            print(f"\n[EARLY STOP] Train loss stagnant for {early_stop_patience} epochs. Halting.")
            break


def run_stage3(arglist, device, model_dir, results_dir):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=results_dir)

    model     = FlowMatchingVLA(arglist).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arglist.lr, weight_decay=1e-4)
    start_epoch = 0

    if arglist.ckpt:
        ckpt_path = os.path.join(model_dir, arglist.ckpt)
        if os.path.exists(ckpt_path):
            print(f"Resuming Stage 3 from {ckpt_path}")
            checkpoint  = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1

    train_data = Stage3Dataset(arglist, "train")
    test_data  = Stage3Dataset(arglist, "test")

    num_workers = 2 if torch.cuda.is_available() else 0
    pin_memory  = torch.cuda.is_available()
    train_loader = DataLoader(train_data, batch_size=arglist.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=stage3_collate_fn)
    test_loader  = DataLoader(test_data,  batch_size=arglist.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=stage3_collate_fn)
    print("Stage 3 data loaded.")

    best_test_loss = np.inf

    for epoch in range(start_epoch, arglist.epochs):
        print(f"\nEpoch {epoch + 1} / {arglist.epochs}")

        # Train 
        model.train()
        train_losses = []
        for O, A in tqdm(train_loader, desc="Training"):
            for k in O:
                O[k] = O[k].to(device)
            A = A.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = model.loss(O, A)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        print(f"Train Loss (MSE): {train_loss:.6f}")
        writer.add_scalar('train_loss', train_loss, epoch)

        # Eval
        model.eval()
        test_losses = []
        with torch.no_grad():
            for O, A in tqdm(test_loader, desc="Testing"):
                for k in O:
                    O[k] = O[k].to(device)
                A = A.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss = model.loss(O, A)
                test_losses.append(loss.item())
        test_loss = np.mean(test_losses)
        print(f"Test  Loss (MSE): {test_loss:.6f}")
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if test_loss < best_test_loss:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                       os.path.join(model_dir, "best.ckpt"))
            best_test_loss = test_loss
            print(">>> New best checkpoint saved!")

        if epoch % 50 == 0 or epoch == arglist.epochs - 1:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                       os.path.join(model_dir, f"{epoch}.ckpt"))

    writer.close()
    print("\nStage 3 training complete!")


def main():
    arglist = parse_args()

    np.random.seed(arglist.seed)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(arglist.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Stage: {arglist.stage} | Experiment: {arglist.expt}")

    model_dir   = os.path.join("/content/drive/MyDrive/APLDL/models/",  arglist.expt)
    results_dir = os.path.join("/content/drive/MyDrive/APLDL/results/", arglist.expt)
    os.makedirs(model_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    runners = {1: run_stage1, 2: run_stage2, 3: run_stage3}
    runners[arglist.stage](arglist, device, model_dir, results_dir)


if __name__ == '__main__':
    main()