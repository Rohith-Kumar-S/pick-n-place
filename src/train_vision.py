import os
import argparse
import numpy as np
import torch
import sys
from src.model import compute_croco_loss, CroCoAutoencoder, visualize_croco_predictions
from src.utils import normalize, get_tensor, create_patch_mask, apply_patch_mask
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from pathlib import Path
import torchvision.transforms as T
from PIL import Image

class CroCoPairDataset(Dataset):
    """
    Loads (topdown_masked, topdown_full) pairs for CroCo-style pretraining.

    By default uses ON-THE-FLY masking from the saved full topdown image so
    each training epoch sees different mask patterns (better generalisation).
    Set use_saved_masks=True to use the pre-computed masks from the NPZ instead
    (useful for exact reproducibility during debugging).s

    Args:
        data_root       : directory containing ep_*.npz files
        patch_size      : must match your ViT patch size (default 16)
        mask_ratio      : fraction of patches to mask at train time (default 0.90)
        img_size        : resize target — must be divisible by patch_size
        use_saved_masks : if True, load topdown_masked from NPZ instead of re-masking
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

        # Build flat index: list of (npz_path, frame_index_within_episode)
        self.samples: list[tuple[Path, int]] = []
        npz_files = sorted(Path(data_root).glob("Copy of ep_*.npz"))
        total_npz_files = len(npz_files)
        print(f"Loading {total_npz_files} NPZ files...")
        # if self.is_train:
        #     npz_files = npz_files[:int(0.8 * total_npz_files)]
        #     npz_files = np.random.permutation(npz_files)  # shuffle training files for better generalisation
        # else:
        #     npz_files = npz_files[int(0.8 * total_npz_files):]
        for npz in npz_files:
            n = np.load(npz, allow_pickle=True)['topdown'].shape[0]
            for i in range(n):
                self.samples.append((npz, i))

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        npz_path, frame_idx = self.samples[idx]
        ep = np.load(npz_path, allow_pickle=True)
        
        # PIL RGB
        topdown_full = Image.fromarray(np.transpose(ep['topdown'][frame_idx],(1,2,0)))   
        gripper_full = Image.fromarray(np.transpose(ep['gripperpov'][frame_idx],(1,2,0))) 
        
        # Apply shared transform (resize + normalise)
        topdown_full_t  = self.transform(topdown_full)    # (3, 224, 224)
        gripper_full_t  = self.transform(gripper_full)    # (3, 224, 224)
        bbox_key = 'bboxes_top' 
        bboxes = ep[bbox_key][frame_idx]
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
        
        return topdown_full_t, gripper_full_t, bboxes_tensor


def check(data_loader, model):
    O, A = next(iter(data_loader))
    for key in O:
        print(key, O[key].size())
    print("action", A.size())
    loss = model.loss(O, A)
    print("loss", loss)
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser("Flow matching")
    parser.add_argument("--env", type=str, default="bin-picking-v3", help="")
    parser.add_argument("--expt", type=str, default="expt_2", help="expt name")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    # Simulation parameters
    parser.add_argument("--d-proprio", type=int, default=4, help="proprio dimension, expt_1: 11, expt2: 4")
    parser.add_argument("--d-act", type=int, default=4, help="action dimension is 4 across meta-world tasks")
    parser.add_argument("--image", action=argparse.BooleanOptionalAction, default=False, help="expt_1: False, expt_2: True")
    parser.add_argument("--camera-id", type=int, default=6, help="6: gripper pov")
    parser.add_argument("--image-height", type=int, default=240, help="image height")
    parser.add_argument("--image-width", type=int, default=240, help="image width")
    parser.add_argument("--text", action="store_true", default=False)
    # Training parameters
    parser.add_argument("--T-flow", type=int, default=20, help="flow time steps for sampling")
    parser.add_argument("--d-model", type=int, default=128, help="hidden size dim, expt_1: 64, expt_2: 128")
    parser.add_argument("--d-emb", type=int, default=32, help="embedding dim (only for expt_2 currently)")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train, expt_1: 100, expt_2: 500")
    parser.add_argument("--evaluate-agent", action="store_true", default=False, help="evaluate agent performance periodically")
    parser.add_argument("--use_backbone", action="store_true", default=False, help="use backbone for image encoding")
    parser.add_argument("--ckpt", type=str, default="best.ckpt")
    parser.add_argument("--num_layers", type=int, default=4, help="number of layers in the MLP")
    parser.add_argument("--num_objects", type=int, default=3, help="number of objects in the scene")
    return parser.parse_args()

def main():
    arglist = parse_args()

    np.random.seed(arglist.seed)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(arglist.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.path.join("/content/drive/MyDrive/APLDL/models/", arglist.expt)
    results_dir = os.path.join("/content/drive/MyDrive/APLDL/results/", arglist.expt)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    # writer = SummaryWriter(log_dir=results_dir)

    model = CroCoAutoencoder().to(device)
    
    checkpoint_path = os.path.join(model_dir, arglist.ckpt)

    if os.path.exists(checkpoint_path):
        # =====================================================================
        # SCENARIO A: RESUMING STAGE 2 (Loading the Whole Model)
        # =====================================================================
        print(f"\n[RESUME] Found existing Stage 2 checkpoint at {checkpoint_path}")
        checkpoint_whole = torch.load(checkpoint_path, map_location=device)
        
        # 1. Load the ENTIRE model (Encoder + trained Decoder)
        model.load_state_dict(checkpoint_whole['model'])
        
        # 2. We STILL must freeze the encoder, otherwise PyTorch unfreezes it on load!
        for name, param in model.named_parameters():
            if 'patch_embed' in name or 'encoder' in name:
                param.requires_grad = False
        print("[RESUME] Stem and Encoder weights kept frozen.")
        
        # 3. Load optimizer state (Filtering for unfrozen params to save memory)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.05)
        optimizer.load_state_dict(checkpoint_whole['optimizer'])
        
        # 4. Resume epoch
        start_epoch = checkpoint_whole['epoch'] + 1
        print(f"[RESUME] Resuming training from Epoch {start_epoch}")

    else:
        # =====================================================================
        # SCENARIO B: CURRICULUM HANDOFF (Starting Stage 2 for the first time)
        # =====================================================================
        print(f"\n[START] No Stage 2 checkpoint found. Initializing Curriculum Handoff...")
        stage1_ckpt_path = "/content/drive/MyDrive/APLDL/models/expt_5/stage1_best.ckpt" 
        
        if os.path.exists(stage1_ckpt_path):
            checkpoint = torch.load(stage1_ckpt_path, map_location=device)
            stage1_weights = checkpoint['model']
            croco_state = model.state_dict()
            
            transfer_dict = {}
            for key, weight in stage1_weights.items():
                if 'decoder' in key:
                    continue
                if key in croco_state and croco_state[key].shape == weight.shape:
                    transfer_dict[key] = weight
                    
            croco_state.update(transfer_dict)
            model.load_state_dict(croco_state) 
            print(f"[CURRICULUM] Successfully injected {len(transfer_dict)} parameter tensors!")
        else:
            print(f"[WARNING] Stage 1 checkpoint not found. Initializing CroCo entirely from scratch.")
            
        # Freeze weights
        for name, param in model.named_parameters():
            if 'patch_embed' in name or 'encoder' in name:
                param.requires_grad = False
        print("[CURRICULUM] Stem and Encoder weights frozen. Training Decoder only.")
        
        # Fresh Optimizer & Epoch
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.05)
        start_epoch = 0
    # start_epoch = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
        )
    train_data = CroCoPairDataset("/content/drive/MyDrive/APLDL/new_data_2/raw/expt_5/", is_train=True)
    test_data = CroCoPairDataset("/content/drive/MyDrive/APLDL/new_data_2/raw/test/", is_train=False)

    if torch.cuda.is_available():
        num_workers=2
        pin_memory=True
    else:
        num_workers=0
        pin_memory=False
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=arglist.batch_size, shuffle=True, 
                                               num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=arglist.batch_size, shuffle=True, 
                                              num_workers=num_workers, pin_memory=pin_memory)
    print("Data loaded")
    # check(train_loader, model)

    # Loop over epochs
    best_test_loss = np.inf
    
    # ==========================================
    # EARLY STOPPING TRACKERS
    # ==========================================
    best_train_loss = np.inf
    early_stop_patience = 4
    train_patience_counter = 0

    for epoch in range(start_epoch, arglist.epochs):
        print("Epoch ", epoch + 1, "/", arglist.epochs)
        
        # Training
        model.train()
        train_loss = []
        for full_topdown, gripper_pov, bboxes_tensor in tqdm(train_loader, total=len(train_loader), desc="Training"):
            full_topdown = full_topdown.to(device)
            gripper_pov = gripper_pov.to(device)
            bboxes_tensor = bboxes_tensor.to(device)
            
            optimizer.zero_grad()

            loss = compute_croco_loss(model, full_topdown, gripper_pov, bboxes_tensor, top_mask_ratio=0.95, grip_mask_ratio=0.40)

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

            train_loss.append(loss.item())
    
        train_loss = np.array(train_loss).mean()

        # ==========================================
        # TRAINING LOSS PATIENCE CHECK
        # ==========================================
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            train_patience_counter = 0 # Reset patience if we hit a new best
        else:
            train_patience_counter += 1
            print(f"-> Train loss plateau. Patience: {train_patience_counter}/{early_stop_patience}")

        # Testing
        with torch.no_grad():
            test_loss = []
            for  full_topdown, gripper_pov, bboxes_tensor in tqdm(test_loader, total=len(test_loader), desc="Testing"):
                full_topdown = full_topdown.to(device)
                gripper_pov = gripper_pov.to(device)
                bboxes_tensor = bboxes_tensor.to(device)
                
                loss = compute_croco_loss(model, full_topdown, gripper_pov, bboxes_tensor, top_mask_ratio=0.95, grip_mask_ratio=0.40)
                test_loss.append(loss.item())
                
        test_loss = np.array(test_loss).mean()
        print("train loss: ", train_loss, "test loss: ", test_loss)
        
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            torch.save({'model' : model.state_dict(),
                        'optimizer' : optimizer.state_dict(), 
                        'epoch' : epoch}, os.path.join(model_dir, "best.ckpt"))
            best_test_loss = test_loss
        
        if epoch % 5 == 0 or epoch == arglist.epochs-1:
            torch.save({'model' : model.state_dict(),
                        'optimizer' : optimizer.state_dict(), 
                        'epoch' : epoch}, os.path.join(model_dir, str(epoch)+".ckpt"))
            print("Visualizing Reconstructions...")
            visualize_croco_predictions(model, full_topdown, gripper_pov, top_mask_ratio = 0.95, grip_mask_ratio=0.40, num_samples=3, save_path=os.path.join(results_dir, f"stage2_epoch_{epoch}.png"))

        # ==========================================
        # EARLY STOPPING TRIGGER
        # ==========================================
        if train_patience_counter >= early_stop_patience:
            print(f"\n[EARLY STOP] Training loss has not improved for {early_stop_patience} consecutive epochs.")
            print("Halting training to save compute!")
            break

    # writer.close()

if __name__ == '__main__':
    main()
