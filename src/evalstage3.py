import os
import time
import argparse
import numpy as np
import torch
import torchvision.transforms as T
import gymnasium as gym
import metaworld
import imageio
from PIL import Image, ImageDraw

# Ensure these match the class names in your updated src/model.py
from src.model import FlowMatchingVLA, CroCoAutoencoder 
from src.utils import check_success

def parse_args():
    parser = argparse.ArgumentParser("Flow Matching VLA Evaluation")
    parser.add_argument("--env", type=str, default="bin-picking-three-objects-v3")
    parser.add_argument("--expt", type=str, default="stage3_main", help="expt name")
    parser.add_argument("--seed", type=int, default=153)
    parser.add_argument("--ckpt", type=str, default="best.ckpt")
    parser.add_argument("--episodes", type=int, default=5)
    
    # Simulation parameters
    parser.add_argument("--d-proprio", type=int, default=4, help="proprio dimension")
    parser.add_argument("--d-act", type=int, default=4, help="action dimension")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-id", type=int, default=6, help="6: gripper pov")
    parser.add_argument("--image-height", type=int, default=224, help="image height")
    parser.add_argument("--image-width", type=int, default=224, help="image width")
    parser.add_argument("--text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_objects", type=int, default=3, help="number of objects")

    # Model parameters
    parser.add_argument("--T-flow", type=int, default=20, help="flow time steps for sampling")
    parser.add_argument("--d-model", type=int, default=256, help="hidden size dim for MLP")
    parser.add_argument("--d-emb", type=int, default=256, help="embedding projection dim")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_layers", type=int, default=4, help="number of layers in the model")
    return parser.parse_args()


def eval_model(arglist):
    np.random.seed(arglist.seed)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(arglist.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # ==========================================
    # 1. INITIALIZE ENVIRONMENTS
    # ==========================================
    # We strictly use "rgb_array" so we can feed the cameras to CroCo
    render_mode = "rgb_array"
    
    env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, 
                   render_mode=render_mode, camera_name="behindGripper",
                   height=arglist.image_height, width=arglist.image_width)
                   
    env_top = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed,
                       render_mode=render_mode, camera_name="topview",
                       height=arglist.image_height, width=arglist.image_width)

    # ==========================================
    # 2. LOAD FROZEN STAGE 2 CROCO
    # ==========================================
    print("Loading Stage 2 CroCo Vision Model...")
    croco_model = CroCoAutoencoder().to(device)
    # Point this strictly to your Stage 2 training output
    croco_ckpt_path = "/content/drive/MyDrive/APLDL/models/expt_6/best.ckpt" 
    croco_ckpt = torch.load(croco_ckpt_path, map_location=device)
    croco_model.load_state_dict(croco_ckpt['model'])
    croco_model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ==========================================
    # 3. LOAD STAGE 3 FLOW MATCHING VLA
    # ==========================================
    model_dir = os.path.join("/content/drive/MyDrive/APLDL/models/", arglist.expt)
    checkpoint_path = os.path.join(model_dir, arglist.ckpt)
    print(f"Loading Stage 3 VLA Model from: {checkpoint_path}")
    
    model = FlowMatchingVLA(arglist).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # ==========================================
    # 4. EVALUATION LOOP
    # ==========================================
    colors = ["green", "yellow", "purple"]
    metric = []
    
    for episode in range(arglist.episodes):
        o, info = env.reset()
        env_top.reset()
        frames = []
        step = 0
        
        target = np.random.choice(arglist.num_objects) if arglist.text else 0
        print(f"\n--- Starting Episode {episode + 1} --- Target: {colors[target].upper()}")

        while True:
            # 4a. Render Camera Feeds
            # Using .copy() to prevent PyTorch negative stride errors
            gripper_img = env.render().copy()
            top_img = env_top.render().copy()
            
            # Combine images side-by-side
            combined_frame = np.concatenate([top_img, gripper_img], axis=1)
            
            # --- Draw the Overlay using PIL ---
            pil_img = Image.fromarray(combined_frame)
            draw = ImageDraw.Draw(pil_img)
            
            overlay_text = f"Stage 3 Inference | Target: {colors[target].upper()}"
            
            # Draw a black rectangle background for readability
            draw.rectangle([(10, 10), (250, 30)], fill="black")
            draw.text((15, 15), overlay_text, fill="white")
            
            frames.append(np.array(pil_img))

            # 4b. Live Feature Extraction
            with torch.no_grad():
                # Process Images
                t_grip = transform(gripper_img).unsqueeze(0).to(device)
                t_top = transform(top_img).unsqueeze(0).to(device)
                
                croco_emb = croco_model(
                    t_top, t_grip, 
                    top_mask_ratio=0.0, 
                    grip_mask_ratio=0.0, 
                    return_embedding_only=True
                )
                
                # Process Proprioception
                proprio_raw = o[:arglist.d_proprio].astype(np.float32)
                if arglist.normalize:
                    proprio_norm = (proprio_raw - model.stats['proprio_mean']) / model.stats['proprio_std']
                else:
                    proprio_norm = proprio_raw
                
                proprio_t = torch.tensor(proprio_norm, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Process Text Target
                text_t = torch.tensor([target], dtype=torch.long).to(device)
                
                # Build Input Dictionary
                O_dict = {
                    'proprio': proprio_t,
                    'croco_embedding': croco_emb,
                    'text': text_t
                }

                # 4c. Flow Matching Action Inference
                a = model.sample(O_dict, device)

            # 4d. Step the Environment
            o_1, r, terminated, truncated, info = env.step(a)
            env_top.step(a)
            
            step += 1
            success = check_success(o_1, target, arglist)
            done = terminated or truncated or success
            o = o_1
            
            if done or step >= 500: # Hard cutoff to prevent infinite loops
                result_str = "SUCCESS" if success else "FAILED"
                print(f"Episode {episode + 1} finished in {step} steps. Result: {result_str}")
                metric.append(success)
                break

        # 4e. Save Episode Artifacts
        gif_path = f"/content/episode_{episode + 1}_{colors[target]}.gif"
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"Saved GIF: {gif_path}")

    print(f"\n======================================")
    print(f"Final Task Success Rate: {np.mean(metric):.2%}")
    print(f"======================================")
    env.close()

if __name__ == '__main__':
    arglist = parse_args()
    eval_model(arglist)