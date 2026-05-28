import os
import numpy as np
import torch
from PIL import Image
# import mujoco
import torch.nn.functional as F
def construct_observation_tensor(o, env, env_top, arglist, stats, device, target=None):
    if arglist.image:
        # In proprio we store only end-effector position and gripper state
        proprio = o[:4]
    else:
        # In proprio we store end-effector position, gripper state,  
        # object position, object orientation
        proprio = o[:11]
    if arglist.normalize:
        O = {'proprio': get_tensor(normalize(proprio, stats['proprio_mean'], stats['proprio_std'])).unsqueeze(0).to(device)}
    else:
        O = {'proprio': get_tensor(proprio).unsqueeze(0).to(device)}

    if arglist.image:
        # Object position, object orientation must be inferred from rgb and depth images 
        rgb_array, depth_array, top_rgb = get_images(env, env_top)
        if arglist.normalize:
            rgb  = get_tensor(normalize(rgb_array.astype(np.float32), stats['rgb_mean'], stats['rgb_std']))
            depth = get_tensor(normalize(depth_array, stats['depth_mean'], stats['depth_std']))

            rgb = F.interpolate(rgb.unsqueeze(0), size=(128,128), mode='bilinear', align_corners=False)
            depth = F.interpolate(depth.unsqueeze(0), size=(128,128), mode='nearest')
            O['rgb'] = rgb.to(device)
            O['depth'] = depth.to(device)
            
            O['topdown'] = [get_tensor(normalize(top_rgb, stats['topdown_mean'], stats['topdown_std'])).to(device)]
            O["target"] = None
        else:
            rgb = get_tensor(rgb_array).float()          
            rgb = F.interpolate(rgb.unsqueeze(0),        
                                size=(128,128),
                                mode='bilinear',
                                align_corners=False)

            depth = get_tensor(depth_array).float()     
            depth = F.interpolate(depth.unsqueeze(0),    
                                size=(128,128),
                                mode='nearest')

            O['rgb'] = rgb.to(device)      
            O['depth'] = depth.to(device)
            
            O['topdown'] = [get_tensor(top_rgb).unsqueeze(0).to(device)]
            O["target"] = None
     # Text
    if arglist.text:
        O['text'] = get_tensor(np.array([target]),dtype=torch.long).unsqueeze(0).to(device)
    return O

def normalize(x, mean, std):
    return (x-mean)/std

def get_tensor(x, dtype=torch.float32):
    return torch.from_numpy(x).to(dtype=dtype)

def get_expert_policy(arglist):
    if arglist.env == "bin-picking-v3" or arglist.env == "bin-picking-two-objects-v3" or arglist.env == "bin-picking-three-objects-v3":
        from metaworld.policies.sawyer_bin_picking_v3_policy import SawyerBinPickingV3Policy as ExpertPolicy
    return ExpertPolicy()

    return ExpertPolicy()

def print_info(t, o, a, r, o_1, terminated, truncated, info):
    print(f"\nt: {t}"
          f"\nobservation: {o.shape}, action: {a.shape}, reward: {r}, "
          f"new observation: {o_1.shape}, terminated: {terminated}, truncated: {truncated}"
          f"\ninfo: {info}")

def get_images(env, env_top):
    rgb_copy = env.unwrapped.mujoco_renderer.render("rgb_array").copy()
    depth_copy = env.unwrapped.mujoco_renderer.render("depth_array").copy()
    rgb_array   = np.transpose(rgb_copy, (2,0,1))
    depth_array = depth_copy[None,:,:]
    top_rgb = env_top.unwrapped.mujoco_renderer.render("rgb_array").copy()
    top_rgb = np.transpose(top_rgb, (2, 0, 1))
    return rgb_array, depth_array, top_rgb

def log_image(t, rgb_array, depth_array, path):
    rgb_img = Image.fromarray(np.transpose(rgb_array, (1,2,0)))
    rgb_img.save(os.path.join(path, "rgb_"+str(t)+".png"))

    depth_array_processed = ((depth_array[0] - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
    depth_img = Image.fromarray(depth_array_processed)
    depth_img.save(os.path.join(path, "depth_"+str(t)+".png"))
    
    
def world_to_pixel(model, data, cam_name, obj_world_pos, img_h, img_w):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    point_cam = cam_mat @ (obj_world_pos - cam_pos)

    fovy = model.cam_fovy[cam_id]
    f = (img_h / 2) / np.tan(np.deg2rad(fovy / 2))
    px = int(f * -point_cam[0] / point_cam[2] + img_w / 2)
    py = int(f * point_cam[1] / point_cam[2] + img_h / 2)
    return px, py

def make_bbox_from_3d(model, data, cam_name, obj_world_pos, obj_size, img_h, img_w):
    cx, cy, cz = obj_world_pos
    w, h, d = obj_size  # half-extents or full size depending on your convention
    
    # 8 corners of the 3D bounding box
    corners = np.array([
        [cx - w/2, cy - h/2, cz - d/2],
        [cx + w/2, cy - h/2, cz - d/2],
        [cx - w/2, cy + h/2, cz - d/2],
        [cx + w/2, cy + h/2, cz - d/2],
        [cx - w/2, cy - h/2, cz + d/2],
        [cx + w/2, cy - h/2, cz + d/2],
        [cx - w/2, cy + h/2, cz + d/2],
        [cx + w/2, cy + h/2, cz + d/2],
    ])
    
    # Project all corners to pixel space
    pixels = []
    for corner in corners:
        px, py = world_to_pixel(model, data, cam_name, corner, img_h, img_w)
        pixels.append((px, py))
    
    pixels = np.array(pixels)
    x1 = max(pixels[:, 0].min(), 0)
    y1 = max(pixels[:, 1].min(), 0)
    x2 = min(pixels[:, 0].max(), img_w)
    y2 = min(pixels[:, 1].max(), img_h)
    
    return np.array([x1, y1, x2, y2], dtype=np.int32)

def make_pixel_bbox(px, py, half_w, half_h, img_w, img_h):
    x1 = max(px - half_w, 0)
    y1 = max(py - half_h, 0)
    x2 = min(px + half_w, img_w)
    y2 = min(py + half_h, img_h)
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def swap_obs(o, target, arglist):
    # Meta-world's expert policy goes to the first object in the observation by default
    # Hence, when the target is not the first object, we swap the target with the first object
    # in the observation vector
    if target == 0:
        return o
    elif arglist.num_objects == 2 and target == 1:
        new_o = o.copy()
        new_o[4:11] = o[11:18]
        new_o[11:18] = o[4:11]
        new_o[22:29] = o[29:36]
        new_o[29:36] = o[22:29]
        return new_o
    elif arglist.num_objects == 3 and target == 1:
        new_o = o.copy()
        new_o[4:11] = o[11:18]
        new_o[11:18] = o[4:11]
        new_o[29:36] = o[36:43]
        new_o[36:43] = o[29:36]
        return new_o
    elif arglist.num_objects == 3 and target == 2:
        new_o = o.copy()
        new_o[4:11] = o[18:25]
        new_o[18:25] = o[4:11]
        new_o[29:36] = o[43:50]
        new_o[43:50] = o[29:36]
        return new_o
    
def check_success(o_1, target, arglist):
    # Meta-world's info['success'] only checks if the default object (green) reached the goal
    # This does not work when we have multiple objects
    # Hence we define our function, which checks if the actual target object reached the goal
    
    if arglist.num_objects == 1:
        obj_pos = o_1[4:7]   # obj1 position

    elif arglist.num_objects == 2:
        if target == 0:
            obj_pos = o_1[4:7]   # obj1 position
        else:
            obj_pos = o_1[11:14] # obj2 position
    elif arglist.num_objects == 3:
        if target == 0:
            obj_pos = o_1[4:7]   # obj1 position
        elif target == 1:
            obj_pos = o_1[11:14] # obj2 position                    
        elif target == 2:
            obj_pos = o_1[18:21] # obj3 position
    
    goal_pos = o_1[-3:]
    correct_obj_success = float(np.linalg.norm(obj_pos - goal_pos) < 0.05)
    success = int(correct_obj_success)
    return success

def apply_patch_mask(img: np.ndarray, pixel_mask: np.ndarray) -> np.ndarray:
    """
    Zero out masked regions in img.

    Args:
        img         : uint8 RGB image, shape (H, W, 3)
        pixel_mask  : uint8 mask, shape (H, W), 1 = masked

    Returns:
        masked_img  : uint8 RGB image with masked patches set to 0
    """
    masked = img.copy()
    masked[pixel_mask == 1] = 0
    return masked

def create_patch_mask(
    img_h: int,
    img_w: int,
    patch_size: int = 16,
    mask_ratio: float = 0.90,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Returns a boolean mask of shape (img_h, img_w) where True = MASKED (zeroed).

    Divides the image into non-overlapping (patch_size x patch_size) patches and
    randomly selects `mask_ratio` fraction of them to mask.

    Args:
        img_h, img_w : image dimensions (must be divisible by patch_size)
        patch_size   : side length of each square patch (default 16, matches ViT-S/16)
        mask_ratio   : fraction of patches to zero out (default 0.90, matches CroCo)
        rng          : optional numpy Generator for reproducibility

    Returns:
        pixel_mask   : uint8 array (img_h, img_w), 1 = masked, 0 = visible
    """
    if rng is None:
        rng = np.random.default_rng()

    n_h = img_h // patch_size
    n_w = img_w // patch_size
    n_patches = n_h * n_w
    n_masked = int(np.round(mask_ratio * n_patches))

    patch_indices = rng.permutation(n_patches)[:n_masked]

    pixel_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for idx in patch_indices:
        row = (idx // n_w) * patch_size
        col = (idx % n_w) * patch_size
        pixel_mask[row:row + patch_size, col:col + patch_size] = 1

    return pixel_mask  # shape (H, W)