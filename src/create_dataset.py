""""
Rohith Kumar Senthil Kumar
Pick and Place Dataset Creation Script

MetaWorld data collection for CroCo-style self-supervised pretraining.

Saves one NPZ per episode containing:
  - topdown        : full top-down frames,   shape (T, H, W, 3) uint8  — reconstruction TARGET
  - gripperpov     : gripper-POV frames,     shape (T, H, W, 3) uint8  — cross-view reference
  - bboxes_top     : 2-D bboxes projected from top-down camera,     shape (T, N_obj, 4) float32
  - bboxes_gripper : 2-D bboxes projected from gripper-POV camera,  shape (T, N_obj, 4) float32

T = number of sub-sampled timesteps per episode (≤ 20 by default).
Each image is stored in HWC layout (H, W, 3) — the dataset classes apply
np.transpose(..., (2, 0, 1)) or T.ToTensor() to convert to CHW before feeding
the model.
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import mujoco
from tqdm import tqdm
from src.utils import get_expert_policy, get_images, make_bbox_from_3d


# Arg parsing

def parse_args():
    parser = argparse.ArgumentParser("Create CroCo Pretraining Dataset")
    parser.add_argument("--env",              type=str, default="bin-picking-v3")
    parser.add_argument("--expt",             type=str, default="expt_4",  help="Sub-folder name under data_root")
    parser.add_argument("--seed",             type=int, default=0)
    parser.add_argument("--episodes",         type=int, default=500,       help="Total episodes to collect")
    parser.add_argument("--start-episode",    type=int, default=0,         help="Resume from this episode index")
    parser.add_argument("--max-steps",        type=int, default=150,       help="Max env steps per episode")
    parser.add_argument("--samples-per-ep",   type=int, default=20,        help="Timesteps to sub-sample per episode")
    parser.add_argument("--image-height",     type=int, default=224)
    parser.add_argument("--image-width",      type=int, default=224)
    parser.add_argument("--num-objects",      type=int, default=3,         help="Number of movable objects in scene")
    parser.add_argument("--text",             action="store_true", default=False)
    return parser.parse_args()

# Main collection loop
def main(arglist):
    np.random.seed(arglist.seed)

    #  Environments 
    env = gym.make(
        'Meta-World/MT1',
        env_name=arglist.env,
        seed=arglist.seed,
        render_mode="rgb_array",
        camera_name="behindGripper",
        height=arglist.image_height,
        width=arglist.image_width,
    )
    env_top = gym.make(
        'Meta-World/MT1',
        env_name=arglist.env,
        seed=arglist.seed,
        render_mode="rgb_array",
        camera_name="topview",
        height=arglist.image_height,
        width=arglist.image_width,
    )

    policy   = get_expert_policy(arglist)
    data_dir = os.path.join("/content/drive/MyDrive/APLDL/new_data_2/raw", arglist.expt)
    os.makedirs(data_dir, exist_ok=True)

    #  Object registry 
    # (body_name, geom_name, half-extents)  — add/remove objects here
    bin_dims = (0.3,  0.3,  0.155)
    cube_dim = (0.04, 0.04, 0.04)
    OBJECTS = {
        "bin_start": ("bin_start_geom", bin_dims),
        "bin_goal":  ("bin_goal_geom",  bin_dims),
        "obj":       ("obj_geom",        cube_dim),
        "obj2":      ("obj_geom",        cube_dim),
        "obj3":      ("obj_geom",        cube_dim),
    }
    # Trim to the number of objects actually present in this experiment
    OBJECTS = dict(list(OBJECTS.items())[:2 + arglist.num_objects])  # bins + N cubes

    #  Episode loop 
    for episode in tqdm(range(arglist.start_episode, arglist.episodes), desc="Episodes"):

        gripper_frames = []   # list of (H, W, 3) uint8
        topdown_frames = []   # list of (H, W, 3) uint8
        bboxes_top_ep     = []  # list of (N_obj, 4) float32
        bboxes_gripper_ep = []  # list of (N_obj, 4) float32

        o, _ = env_top.reset()
        env.reset()

        target = np.random.randint(arglist.num_objects) if arglist.text else 0
        step   = 0

        while True:
            # Render
            rgb_array, _, top_rgb = get_images(env, env_top)

            topdown_frames.append(top_rgb.astype(np.uint8))       # HWC
            gripper_frames.append(rgb_array.astype(np.uint8))     # HWC

            # Bounding boxes (MuJoCo 3-D → 2-D pixel projection) 
            mj_model = env_top.unwrapped.model
            mj_data  = env_top.unwrapped.data

            step_bboxes_top     = []
            step_bboxes_gripper = []

            for body_name, (geom_name, obj_size) in OBJECTS.items():
                body_id      = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                obj_world_pos = mj_data.xpos[body_id]

                bbox_top = make_bbox_from_3d(
                    mj_model, mj_data, "topview",
                    obj_world_pos, obj_size,
                    arglist.image_height, arglist.image_width,
                )
                bbox_grip = make_bbox_from_3d(
                    mj_model, mj_data, "behindGripper",
                    obj_world_pos, obj_size,
                    arglist.image_height, arglist.image_width,
                )

                step_bboxes_top.append(bbox_top)
                step_bboxes_gripper.append(bbox_grip)

            bboxes_top_ep.append(np.array(step_bboxes_top,     dtype=np.float32))   # (N_obj, 4)
            bboxes_gripper_ep.append(np.array(step_bboxes_gripper, dtype=np.float32))

            # Step 
            a = policy.get_action(o)
            o, _, terminated, truncated, info = env_top.step(a)
            env.step(a)
            step += 1

            done = terminated or truncated or bool(info.get('success', False))
            if step >= arglist.max_steps or done:
                # Sub-sample timesteps 
                n_samples = min(step, arglist.samples_per_ep)
                indices   = np.sort(np.random.choice(step, size=n_samples, replace=False))

                np.savez_compressed(
                    os.path.join(data_dir, f"ep_{episode}.npz"),
                    # Images stored as HWC uint8 — transpose to CHW in the Dataset class
                    topdown         = np.array([topdown_frames[i]      for i in indices], dtype=np.uint8),
                    gripperpov      = np.array([gripper_frames[i]      for i in indices], dtype=np.uint8),
                    bboxes_top      = np.array([bboxes_top_ep[i]       for i in indices], dtype=np.float32),
                    bboxes_gripper  = np.array([bboxes_gripper_ep[i]   for i in indices], dtype=np.float32),
                )
                print(f"  ep {episode:04d} | steps: {step} | sampled: {n_samples} | target: {target}")
                break

# Sanity check
def simple_check(data_dir: str, n: int = 3):
    """Print shapes of the first n episode files."""
    from pathlib import Path
    files = sorted(Path(data_dir).glob("ep_*.npz"))[:n]
    for f in files:
        ep = np.load(f, allow_pickle=True)
        print(f"\n{f.name}")
        for key in ep:
            print(f"  {key}: dtype={ep[key].dtype}  shape={ep[key].shape}")


if __name__ == '__main__':
    arglist = parse_args()
    main(arglist)
    data_dir = os.path.join("/content/drive/MyDrive/APLDL/new_data_2/raw", arglist.expt)
    simple_check(data_dir)