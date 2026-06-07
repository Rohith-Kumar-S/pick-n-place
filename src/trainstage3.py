import os
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import sys
from src.model import FlowMatchingVLA # Ensure this imports your updated Stage3VectorField
from src.utils import normalize, get_tensor
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    """
    dataset['proprio']: N, 4
    dataset['croco_embedding']: N, 128
    dataset['text']: N (Target ID)
    dataset['action']: N, 4
    """
    def __init__(self, arglist, mode):
        super().__init__()
        self.arglist = arglist
        
        # Base directory
        data_dir = os.path.join("/content/drive/MyDrive/APLDL/data/", "raw")
        
        # Route to correct subfolders
        if mode == "train":
            data_path = os.path.join(data_dir, "train", "train.npz")
        else:
            data_path = os.path.join(data_dir, "test", "test.npz")
            
        # Stats MUST always come from the train folder!
        stats_path = os.path.join(data_dir, "train", "stats.npz")
        
        print(f"Loading {mode} data from: {data_path}")
        dataset = np.load(data_path, allow_pickle=True)
        stats = np.load(stats_path, allow_pickle=True)
        
        self.proprio = dataset['proprio']
        self.A = dataset['action']
        
        if self.arglist.image:
            self.croco_embedding = dataset['croco_embedding']
            
        if self.arglist.text:
            self.text = dataset['text']

        if self.arglist.normalize:
            self.proprio_mean = stats['proprio_mean']
            self.proprio_std = stats['proprio_std']        
            self.A_mean = stats['action_mean']
            self.A_std = stats['action_std']
            
        self.dims = self.A.shape
    
    def __len__(self):
        return self.dims[0]

    def __getitem__(self, idx):
        n = idx

        if self.arglist.normalize:
            o = {'proprio': get_tensor(normalize(self.proprio[n], self.proprio_mean, self.proprio_std))}
            a = get_tensor(normalize(self.A[n], self.A_mean, self.A_std))
        else:
            o = {'proprio': get_tensor(self.proprio[n])}
            a = get_tensor(self.A[n])
            
        if self.arglist.image:
            o['croco_embedding'] = get_tensor(self.croco_embedding[n])
            
        if self.arglist.text:
            # Bypass get_tensor to safely convert the scalar directly
            o['text'] = torch.tensor(self.text[n], dtype=torch.long)
        
        return o, a

def parse_args():
    parser = argparse.ArgumentParser("Stage 3 Flow Matching VLA")
    parser.add_argument("--env", type=str, default="bin-picking-three-objects-v3", help="")
    parser.add_argument("--expt", type=str, default="expt_4", help="expt name")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    
    # Simulation parameters
    parser.add_argument("--d-proprio", type=int, default=4, help="proprio dimension")
    parser.add_argument("--d-act", type=int, default=4, help="action dimension")
    parser.add_argument("--image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text", action=argparse.BooleanOptionalAction, default=True)
    
    # Training parameters
    parser.add_argument("--T-flow", type=int, default=20, help="flow time steps for sampling")
    parser.add_argument("--d-model", type=int, default=256, help="hidden size dim for MLP")
    parser.add_argument("--d-emb", type=int, default=256, help="embedding projection dim")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=256, help="batch size") # Increased since vectors are tiny!
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train")
    parser.add_argument("--num_layers", type=int, default=4, help="number of layers in the MLP")
    parser.add_argument("--num_objects", type=int, default=3, help="number of categorical targets")
    return parser.parse_args()

def collate_fn(batch):
    Os, As = zip(*batch)
    O_out = {}
    
    # Simple stacking since all inputs are 1D vectors now!
    for k in Os[0].keys():
        O_out[k] = torch.stack([o[k] for o in Os])  
    A = torch.stack(As)

    return O_out, A

def main():
    arglist = parse_args()

    np.random.seed(arglist.seed)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(arglist.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model_dir = os.path.join("/content/drive/MyDrive/APLDL/models/", arglist.expt)
    results_dir = os.path.join("/content/drive/MyDrive/APLDL/results/", arglist.expt)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=results_dir)

    # Initialize updated Model
    model = FlowMatchingVLA(arglist).to(device)

    # Single unified optimizer (No CNN backbone to worry about)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    start_epoch = 0
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=5
    # )
    
    train_data = Dataset(arglist, "train")
    test_data = Dataset(arglist, "test")

    num_workers = 2 if torch.cuda.is_available() else 0
    pin_memory = True if torch.cuda.is_available() else False
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=arglist.batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=arglist.batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn
    )
    print("Data loaded successfully.")

    best_test_loss = np.inf
    
    for epoch in range(start_epoch, arglist.epochs):
        print(f"\nEpoch {epoch + 1} / {arglist.epochs}")
        
        # ====================================================
        # TRAINING
        # ====================================================
        model.train()
        train_loss_tracker = []
        
        for O, A in tqdm(train_loader, total=len(train_loader), desc="Training"):
            for k in O:
                O[k] = O[k].to(device)
            A = A.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Returns only the pure Action MSE loss now
                loss = model.loss(O, A)

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

            train_loss_tracker.append(loss.item())
            
        train_loss = np.mean(train_loss_tracker)
        print(f"Train Loss (MSE): {train_loss:.6f}")
        writer.add_scalar('train_loss', train_loss, epoch)

        # ====================================================
        # TESTING
        # ====================================================
        model.eval()
        with torch.no_grad():
            test_loss_tracker = []
            
            for O, A in tqdm(test_loader, total=len(test_loader), desc="Testing"):
                for k in O:
                    O[k] = O[k].to(device)
                A = A.to(device)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss = model.loss(O, A)
                    
                test_loss_tracker.append(loss.item())
                
        test_loss = np.mean(test_loss_tracker)
        print(f"Test Loss (MSE):  {test_loss:.6f}")
        
        # scheduler.step(test_loss)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # ====================================================
        # CHECKPOINTING
        # ====================================================
        if test_loss < best_test_loss:
            torch.save({
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(), 
                'epoch' : epoch
            }, os.path.join(model_dir, "best.ckpt"))
            best_test_loss = test_loss
            print(">>> New Best Checkpoint Saved!")
        
        if epoch % 50 == 0 or epoch == arglist.epochs - 1:
            torch.save({
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(), 
                'epoch' : epoch
            }, os.path.join(model_dir, f"{epoch}.ckpt"))

        # NOTE: Live `evaluate_agent` in Metaworld requires the frozen CroCo model 
        # to generate embeddings on the fly. To save VRAM during training, 
        # it is recommended to evaluate using a separate inference script.

    writer.close()
    print("\nTraining Complete!")

if __name__ == '__main__':
    main()