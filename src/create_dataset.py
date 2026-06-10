""""
Rohith Kumar Senthil Kumar
Pick and Place Dataset Creation Script
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.utils import get_expert_policy, get_images, make_bbox_from_3d
import mujoco

def parse_args():
    parser = argparse.ArgumentParser("Create Dataset")
    parser.add_argument("--env",      type=str,  default="bin-picking-v3")
    parser.add_argument("--expt", type=str, default="expt_2", help="expt name")
    parser.add_argument("--seed",     type=int,  default=0)
    parser.add_argument("--episodes", type=int,  default=100)
    parser.add_argument("--image", action=argparse.BooleanOptionalAction, default=False, help="expt_1: False, expt_2: True")
    parser.add_argument("--camera-id", type=int, default=6, help="6: gripper pov")
    parser.add_argument("--image-height", type=int, default=240, help="image height")
    parser.add_argument("--image-width", type=int, default=240, help="image width")
    parser.add_argument("--text",     action="store_true", default=False)
    return parser.parse_args()

def main(arglist):
    np.random.seed(arglist.seed)

    if arglist.image:
        env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode="rgb_array",\
                        camera_id=arglist.camera_id ,height=arglist.image_height,width=arglist.image_width)
        env_top = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed,
                        render_mode="rgb_array", camera_name="topview",height=arglist.image_height, 
                        width=arglist.image_width)
    else:
        env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode="none")        

    policy = get_expert_policy(arglist)
    bin_dims = (0.2,0.2, 0.155)
    cube_dim = (0.05, 0.05, 0.05)
    OBJECTS = {
    "bin_start": (1, "bin_start_geom", bin_dims),   # label, geom_name
    "bin_goal":  (2, "bin_goal_geom", bin_dims),
    "obj":       (3, "obj_geom", cube_dim),
    }
    proprio, action = [], []
    if arglist.image:
        rgb = []
        depth = []
        topdown = []
        bboxes = []
        labels = []
    # if arglist.text:
    #     T = []
    episode_starts = [0]
    for episode in tqdm(range(arglist.episodes)):
        o, info = env.reset()
        env_top.reset()
        while True:
            if arglist.image:
                # In proprio we store only end-effector position and gripper state
                proprio.append(o[:4].astype(np.float32))
                # Object position, object orientation must be inferred from rgb and depth images 
                rgb_array, depth_array, top_rgb = get_images(env, env_top)
                rgb.append(rgb_array.astype(np.uint8))
                depth.append(depth_array.astype(np.float32))
                topdown.append(top_rgb.astype(np.uint8))
                mj_model = env_top.unwrapped.model
                mj_data = env_top.unwrapped.data
                step_bboxes = []
                step_labels = []
                for body_name, (label_id, geom_name, obj_size) in OBJECTS.items():
                    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                    obj_world_pos = mj_data.xpos[body_id]
                    bbox = make_bbox_from_3d(mj_model, mj_data, "topview", obj_world_pos, obj_size, 
                                            arglist.image_height, arglist.image_width)

                
                    step_bboxes.append(bbox)
                    step_labels.append(label_id)

                bboxes.append(np.array(step_bboxes))   
                labels.append(np.array(step_labels))   
            else:
                # In proprio we store end-effector position, gripper state,  
                # object position, object orientation
                proprio.append(o[:11].astype(np.float32))
            a = policy.get_action(o)
            o_1, r, terminated, truncated, info = env.step(a)
            env_top.step(a)
            success = int(info['success'])
            done = terminated or truncated or success
            action.append(a.astype(np.float32))
            o = o_1
            if done:
                break
        episode_starts.append(len(proprio))

        # if arglist.text:
        #     T.append(env.text_instruction)

    proprio, action = np.array(proprio), np.array(action)
    if arglist.image:
        rgb = np.array(rgb)
        depth = np.array(depth)
        topdown = np.array(topdown)
        bboxes = np.array(bboxes)
        labels = np.array(labels)
    # if arglist.text:
    #     T = np.array(T, dtype=np.dtypes.StringDType())

    episode_indices = np.arange(arglist.episodes)
    train_eps, test_eps = train_test_split(episode_indices, test_size=0.2, random_state=arglist.seed)

    train_indices = np.concatenate([np.arange(episode_starts[e], episode_starts[e+1]) for e in train_eps])
    test_indices = np.concatenate([np.arange(episode_starts[e], episode_starts[e+1]) for e in test_eps])
    # train_indices, test_indices = train_test_split(np.arange(proprio.shape[0]), test_size=0.2, random_state=arglist.seed)
    train_data = {'proprio': proprio[train_indices], 'action': action[train_indices]}
    test_data = {'proprio': proprio[test_indices], 'action': action[test_indices]}
    if arglist.image:
        train_data['rgb'] = rgb[train_indices]
        test_data['rgb'] = rgb[test_indices]

        train_data['depth'] = depth[train_indices]
        test_data['depth'] = depth[test_indices]
        
        train_data['bboxes'] = bboxes[train_indices]
        test_data['bboxes'] = bboxes[test_indices]
        train_data['labels'] = labels[train_indices]
        test_data['labels'] = labels[test_indices]
        
        train_data['topdown'] = topdown[train_indices]
        test_data['topdown'] = topdown[test_indices]

    # if arglist.text:
    #     train_data['text'] = T[train_indices]
    #     test_data['text'] = T[test_indices]

    # 2. Calculate Mean and Std
    stats = {}
    stats['proprio_mean'] = np.mean(train_data['proprio'], axis=0)
    stats['proprio_std']  = np.std(train_data['proprio'], axis=0) + 1e-6 # Avoid division by zero
    
    stats['action_mean'] = np.mean(train_data['action'], axis=0)
    stats['action_std']  = np.std(train_data['action'], axis=0) + 1e-6
    

    if arglist.image:
        stats['rgb_mean'] = train_data['rgb'].astype(np.float32).mean(axis=(0,2,3)).reshape(3,1,1)
        stats['rgb_std']  = train_data['rgb'].astype(np.float32).std(axis=(0,2,3)).reshape(3,1,1)  + 1e-6
        stats['depth_mean'] = train_data['depth'].mean(axis=(0,2,3)).reshape(1,1,1)
        stats['depth_std']  = train_data['depth'].std(axis=(0,2,3)).reshape(1,1,1) + 1e-6
        stats['topdown_mean'] = train_data['topdown'].astype(np.float32).mean(axis=(0,2,3)).reshape(3,1,1)
        stats['topdown_std']  = train_data['topdown'].astype(np.float32).std(axis=(0,2,3)).reshape(3,1,1) + 1e-6

    # 3. Save everything
    data_dir = os.path.join("./data/raw", arglist.expt)
    os.mkdir(data_dir)
    np.savez_compressed(os.path.join(data_dir, "train.npz"), **train_data)
    np.savez_compressed(os.path.join(data_dir, "test.npz"), **test_data)
    np.savez_compressed(os.path.join(data_dir, "stats.npz"), **stats)

    print("Done.")

def simple_check(arglist):
    data_dir = os.path.join("./data/raw", arglist.expt)
    for mode in ["train", "test", "stats"]:
        dataset = np.load(os.path.join(data_dir, mode+".npz"), allow_pickle=True)
        print(f"\n{mode}")
        for key in dataset:
            value = dataset[key]
            print(f"  {key}: dtype: {value.dtype} shape: {value.shape}")
            # if key == 'text':
            #     print(dataset[key])


if __name__ == '__main__':
    arglist = parse_args()
    main(arglist)
    simple_check(arglist)