from collections import defaultdict
import math
import time
from random import normalvariate
from matplotlib import pyplot as plt
from env_cuda import Env
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2
import numpy as np
import sys
sys.path.append('/home/andrewzhang/RAFT/core/')

import argparse
import os
import torchvision
from model import Model
from xception_model import XceptionModel
from raft import RAFT
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights, raft_small, Raft_Small_Weights
from torchvision.utils import flow_to_image
import kornia


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_iters', type=int, default=50000)
parser.add_argument('--coef_v', type=float, default=1.0, help='smooth l1 of norm(v_set - v_real)')
parser.add_argument('--coef_speed', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_v_pred', type=float, default=2.0, help='mse loss for velocity estimation (no odom)')
parser.add_argument('--coef_collide', type=float, default=2.0, help='softplus loss for collision (large if close to obstacle, zero otherwise)')
parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='quadratic clearance loss')
parser.add_argument('--coef_d_acc', type=float, default=0.01, help='control acceleration regularization')
parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='control jerk regularizatinon')
parser.add_argument('--coef_d_snap', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_ground_affinity', type=float, default=0., help='legacy')
parser.add_argument('--coef_bias', type=float, default=0.0, help='legacy')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--cam_angle', type=int, default=10)
parser.add_argument('--single', default=False, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--yaw_drift', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')
parser.add_argument('--use_depth_ratio', default=False, action='store_true')
parser.add_argument('--use_optical_flow', default=False, action='store_true')
parser.add_argument('--ckpt_dir', default='checkpoints')
args = parser.parse_args()
writer = SummaryWriter()
print(args)

device = torch.device('cuda')
env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle)
if args.no_odom:
    model = Model(7, 6, in_channels=3 if args.use_optical_flow else 1)
else:
    model = XceptionModel(7+3, 6, in_channels=3 if args.use_optical_flow else 1)
model = model.to(device)
os.makedirs(args.ckpt_dir, exist_ok=False)

if args.resume:
    state_dict = torch.load(args.resume, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    if missing_keys:
        print("missing_keys:", missing_keys)
    if unexpected_keys:
        print("unexpected_keys:", unexpected_keys)
optim = AdamW(model.parameters(), args.lr, weight_decay=0.01)
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

ctl_dt = 1 / 15


scaler_q = defaultdict(list)
def smooth_dict(ori_dict):
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))

def barrier(x: torch.Tensor, v_to_pt):
    return (v_to_pt * (1 - x).relu().pow(2)).mean()

def is_save_iter(i):
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0

def load_raft_model(checkpoint_path: str, device='cuda', small=False):
    """
    Instantiate RAFT model and load checkpoint. Returns model on device, eval()'d.
    """
    # build args similar to RAFT's examples
    args = argparse.Namespace()
    args.small = small
    args.mixed_precision = False
    args.alternate_corr = False

    model = RAFT(args)
    state = torch.load(checkpoint_path, map_location='cuda')
    state = {k.replace('module.', ''): v for k, v in state.items()}
    # checkpoint sometimes contains "state_dict"
    if 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model

@torch.no_grad()
def compute_flow_raft_batch(
    model,
    transforms,
    depth_prev: torch.Tensor,
    depth_curr: torch.Tensor,
    device: torch.device = "cuda",
    iters: int = 20,
    chunk: int = 4,
    percentile_low: float = 0.02,
    percentile_high: float = 0.98,
    eps: float = 1e-6,
):
    """
    Compute optical flow between depth_prev and depth_curr using an already-loaded RAFT model.
    Inputs:
      - model: RAFT model (already moved to device and set to eval()).
      - depth_prev, depth_curr: (B,1,H,W) torch tensors on any device (will move to device)
    Returns:
      - flow: (B,2,H,W) torch tensor on same device as model, dtype float32
    Notes:
      - This keeps tensors on GPU. Processes in chunks to avoid OOM.
      - We normalize each sample to [0,1] using percentiles (GPU quantile).
      - We repeat the single channel into 3 channels for RAFT input.
    """
    assert depth_prev.shape == depth_curr.shape
    assert depth_prev.ndim == 4 and depth_prev.shape[1] == 1
    if device is None:
        device = next(model.parameters()).device

    # depth_prev = depth_prev.to(device)
    # depth_curr = depth_curr.to(device)
    B, _, H, W = depth_prev.shape

    # Compute per-sample percentiles on GPU using torch.quantile
    # Flatten per-sample
    # flat_prev = depth_prev.view(B, -1)
    # flat_curr = depth_curr.view(B, -1)
    # q_lo = torch.tensor(percentile_low, device=device)
    # q_hi = torch.tensor(percentile_high, device=device)

    # # torch.quantile expects q in [0,1]
    # lo_prev = torch.quantile(flat_prev, q_lo, dim=1).view(B, 1, 1, 1)
    # hi_prev = torch.quantile(flat_prev, q_hi, dim=1).view(B, 1, 1, 1)
    # lo_curr = torch.quantile(flat_curr, q_lo, dim=1).view(B, 1, 1, 1)
    # hi_curr = torch.quantile(flat_curr, q_hi, dim=1).view(B, 1, 1, 1)

    # # normalize each frame to [0,1]
    # denom_prev = (hi_prev - lo_prev).clamp_min(eps)
    # denom_curr = (hi_curr - lo_curr).clamp_min(eps)
    # prev_norm = (depth_prev - lo_prev) / denom_prev
    # curr_norm = (depth_curr - lo_curr) / denom_curr
    # prev_norm = prev_norm.clamp(0.0, 1.0)
    # curr_norm = curr_norm.clamp(0.0, 1.0)

    # replicate channel to 3 channels (RAFT expects 3-channel input)
    img1 = depth_prev.repeat(1, 3, 1, 1)  # (B,3,H,W)
    img2 = depth_curr.repeat(1, 3, 1, 1)

    img1 = img1.to(device)
    img2 = img2.to(device)
    img1_resized = F.interpolate(img1, size=(144, 192), mode="bilinear", align_corners=False, antialias=True)
    img2_resized = F.interpolate(img2, size=(144, 192), mode="bilinear", align_corners=False, antialias=True)
    # img1_resized = torchvision.transforms.functional.resize(img1, size=(144, 192), antialias=True)
    # img2_resized = torchvision.transforms.functional.resize(img2, size=(144, 192), antialias=True)
    img1_transformed, img2_transformed = transforms(img1_resized, img2_resized)
    list_of_flows = model(img1_transformed, img2_transformed)
    predicted_flow = list_of_flows[-1]
    predicted_flow = F.interpolate(predicted_flow, size=(H, W), mode="area")
    # predicted_flow = torchvision.transforms.functional.resize(predicted_flow, size=(H, W), antialias=True)
    return predicted_flow

    # process in chunks
    # for start in range(0, B, chunk):
    #     end = min(B, start + chunk)
    #     i1 = img1[start:end]  # (C,3,H,W)
    #     i2 = img2[start:end]
    #     # RAFT wants inputs in float [0,1] (many implementations), and on the same device
    #     i1 = i1.to(device)
    #     i2 = i2.to(device)

    #     i1_resized = F.resize(i1, size=(256, 256), antialias=True)
    #     i2_resized = F.resize(i2, size=(256, 256), antialias=True)
    #     i1_transformed = transforms(i1_resized)
    #     i2_transformed = transforms(i2_resized)
    #     # model forward: some RAFT implementations return tuple (flow_low, flow_up) or flow_up directly
    #     model_out = model(i1_transformed, i2_transformed)
    #     # Extract final upsampled flow in a robust way:
    #     if isinstance(model_out, (tuple, list)):
    #         flow_chunk = model_out[-1]
    #     else:
    #         flow_chunk = model_out
    #     # ensure shape is (N,2,H,W)
    #     if flow_chunk.ndim == 5:
    #         # some implementations return flow with an extra dimension - handle last
    #         flow_chunk = flow_chunk.squeeze(0)
    #     # move to float32 and append
    #     out_flows.append(flow_chunk.detach().float().to(device))
    # flow = torch.cat(out_flows, dim=0)  # (B,2,H,W)
    # print(flow.shape)
    # return flow

def compute_flow_farneback_batch(depth_prev_t: torch.Tensor,
                                 depth_t: torch.Tensor,
                                 pyr_scale=0.5,
                                 levels=3,
                                 winsize=15,
                                 iterations=3,
                                 poly_n=5,
                                 poly_sigma=1.2,
                                 flags=0,
                                 normalize=True,
                                 median_blur_ksize=5,
                                 percentile_low=2.0,
                                 percentile_high=98.0):
    """
    Compute dense optical flow (Farneback) between depth_prev_t and depth_t.
    Inputs:
      - depth_prev_t, depth_t: torch.Tensor shape (B, 1, H, W), dtype float, positive depths.
    Returns:
      - flow: torch.FloatTensor shape (B, 2, H, W) on same device as inputs, channels (flow_x, flow_y) in pixels.
    Notes:
      - This runs OpenCV Farneback on CPU. If tensors are on GPU they are moved to CPU internally.
      - Invalid depths (<=0 or non-finite) are masked; flow is set to 0 there.
    """
    assert depth_prev_t.ndim == 4 and depth_prev_t.shape[1] == 1
    assert depth_prev_t.shape == depth_t.shape
    device = depth_t.device
    B, _, H, W = depth_t.shape

    # move to cpu numpy arrays (OpenCV is run on CPU here)
    dp_cpu = depth_prev_t.detach().to('cpu').numpy()[:, 0]  # (B, H, W)
    d_cpu  = depth_t.detach().to('cpu').numpy()[:, 0]       # (B, H, W)

    flows = np.zeros((B, 2, H, W), dtype=np.float32)

    for i in range(B):
        img1 = dp_cpu[i]
        img2 = d_cpu[i]

        # validity mask
        mask_valid = np.isfinite(img1) & (img1 < 24) & np.isfinite(img2) & (img2 < 24)

        # If no valid pixels, set zero flow
        if not mask_valid.any():
            continue

        # Normalize per-frame using percentiles to 0..255 uint8 (robust to outliers)
        if normalize:
            lo1 = np.percentile(img1[mask_valid], percentile_low)
            hi1 = np.percentile(img1[mask_valid], percentile_high)
            if hi1 <= lo1:
                hi1 = lo1 + 1e-3
            im1 = np.clip((img1 - lo1) / (hi1 - lo1), 0.0, 1.0)

            lo2 = np.percentile(img2[mask_valid], percentile_low)
            hi2 = np.percentile(img2[mask_valid], percentile_high)
            if hi2 <= lo2:
                hi2 = lo2 + 1e-3
            im2 = np.clip((img2 - lo2) / (hi2 - lo2), 0.0, 1.0)

            im1_u8 = (im1 * 255.0).astype(np.uint8)
            im2_u8 = (im2 * 255.0).astype(np.uint8)
        else:
            # use float32 images (Farneback accepts float32), but uint8 historically more stable
            im1_u8 = img1.astype(np.float32)
            im2_u8 = img2.astype(np.float32)

        # Optional median smoothing to remove single-pixel spikes
        if median_blur_ksize and median_blur_ksize > 1:
            k = median_blur_ksize if median_blur_ksize % 2 == 1 else median_blur_ksize + 1
            try:
                im1_u8 = cv2.medianBlur(im1_u8, k)
                im2_u8 = cv2.medianBlur(im2_u8, k)
            except Exception:
                # If using float32 images medianBlur may fail; ignore smoothing in that case
                pass

        # Compute Farneback flow (returns H x W x 2 float32)
        flow = cv2.calcOpticalFlowFarneback(im1_u8, im2_u8,
                                            None,
                                            pyr_scale,
                                            levels,
                                            winsize,
                                            iterations,
                                            poly_n,
                                            poly_sigma,
                                            flags)

        # hsv = np.zeros((H, W, 3), dtype=np.uint8)
        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[..., 0] = ang*180/np.pi/2
        # hsv[..., 1] = 255
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imwrite(f'flow_vis/frame_{i}.png', bgr)

        flows[i, 0] = flow[..., 0].astype(np.float32)  # x displacement (cols)
        flows[i, 1] = flow[..., 1].astype(np.float32)  # y displacement (rows)

        # Zero-out flow on invalid pixels
        if not mask_valid.all():
            inv_mask = ~mask_valid
            flows[i, :, inv_mask] = 0.0

    # return torch tensor on original device
    return torch.from_numpy(flows).to(device)  # (B,2,H,W)

pbar = tqdm(range(args.num_iters), ncols=80)
# depths = []
# states = []
B = args.batch_size
for i in pbar:
    env.reset()
    model.reset()
    p_history = []
    v_history = []
    target_v_history = []
    vec_to_pt_history = []
    act_diff_history = []
    v_preds = []
    vid = []
    v_net_feats = []
    h = None

    act_lag = 1
    act_buffer = [env.act] * (act_lag + 1)
    target_v_raw = env.p_target - env.p
    if args.yaw_drift:
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
        zeros = torch.zeros_like(drift_av)
        ones = torch.ones_like(drift_av)
        R_drift = torch.stack([
            torch.cos(drift_av), -torch.sin(drift_av), zeros,
            torch.sin(drift_av), torch.cos(drift_av), zeros,
            zeros, zeros, ones,
        ], -1).reshape(B, 3, 3)

    if args.use_depth_ratio:
        depth_history = torch.ones((B, 48, 64), device=device)
        # raft_model = load_raft_model('/home/andrewzhang/RAFT/models/raft-sintel.pth', device=device, small=False)
        # raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device).eval()
        # transforms = Raft_Small_Weights.DEFAULT.transforms()

    for t in range(args.timesteps):
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)
        depth, flow = env.render(ctl_dt)
        p_history.append(env.p)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())

        if is_save_iter(i):
            vid.append(depth[4])

        if args.yaw_drift:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()
        env.run(act_buffer[t], ctl_dt, target_v_raw)

        R = env.R
        fwd = env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd = F.normalize(fwd, 2, -1)
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)

        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)
        state = [
            torch.squeeze(target_v[:, None] @ R, 1),
            env.R[:, 2],
            env.margin[:, None]]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom:
            state.insert(0, local_v)
        state = torch.cat(state, -1)

        # normalize
        depth[depth == 0] = 24  # 24 meter barrier for uncertain pixels
        if args.use_depth_ratio:
            # Calculate depth ratio instead of using raw depth
            # est_optical_flow = compute_flow_raft_batch(raft_model, transforms, depth_history[:, None], depth[:, None], device=device, iters=20, chunk=4, percentile_low=0.02, percentile_high=0.98, eps=1e-6)
            with torch.no_grad():
                depth_ratio = torch.log(depth) - torch.log(depth_history)
                # depth_ratio = depth / depth_history
                depth_history = depth.clone()
                x = depth_ratio.clamp_(-1.0, 1.0) + torch.randn_like(depth_ratio) * 0.02
                # x = (1/15) / (1 - depth_ratio + 1e-6)
                # x = x.clamp_(0.0, 10.0)
                x = x[:, None]
                # x = kornia.filters.bilateral_blur(input=x, kernel_size=(3, 3), sigma_color=0.1, sigma_space=(1.5, 1.5))
                # x = torch.cat([x[:, None], est_optical_flow], dim=1)
                x = F.avg_pool2d(x, 4, 4)
        else:
            x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
            x = F.max_pool2d(x[:, None], 4, 4)

        act, _, h = model(x, state, h)

        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        v_preds.append(v_pred)
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)
        v_net_feats.append(torch.cat([act, local_v, h], -1))

        v_history.append(env.v)
        target_v_history.append(target_v)
        
    p_history = torch.stack(p_history)
    loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()
    act_buffer = torch.stack(act_buffer)

    v_history = torch.stack(v_history)
    v_history_cum = v_history.cumsum(0)
    v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30
    target_v_history = torch.stack(target_v_history)
    T, B, _ = v_history.shape
    delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

    v_preds = torch.stack(v_preds)
    loss_v_pred = F.mse_loss(v_preds, v_history.detach())

    target_v_history_norm = torch.norm(target_v_history, 2, -1)
    target_v_history_normalized = target_v_history / target_v_history_norm[..., None]
    fwd_v = torch.sum(v_history * target_v_history_normalized, -1)
    loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3

    jerk_history = act_buffer.diff(1, 0).mul(15)
    snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
    loss_d_acc = act_buffer.pow(2).sum(-1).mean()
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean()
    loss_d_snap = snap_history.pow(2).sum(-1).mean()

    vec_to_pt_history = torch.stack(vec_to_pt_history)
    distance = torch.norm(vec_to_pt_history, 2, -1)
    distance = distance - env.margin
    with torch.no_grad():
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
    loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()

    speed_history = v_history.norm(2, -1)
    loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm)

    loss = args.coef_v * loss_v + \
        args.coef_obj_avoidance * loss_obj_avoidance + \
        args.coef_bias * loss_bias + \
        args.coef_d_acc * loss_d_acc + \
        args.coef_d_jerk * loss_d_jerk + \
        args.coef_d_snap * loss_d_snap + \
        args.coef_speed * loss_speed + \
        args.coef_v_pred * loss_v_pred + \
        args.coef_collide * loss_collide + \
        args.coef_ground_affinity * loss_ground_affinity

    if torch.isnan(loss):
        print("loss is nan, exiting...")
        exit(1)

    pbar.set_description_str(f'loss: {loss:.3f}')
    optim.zero_grad()
    loss.backward()
    optim.step()
    sched.step()


    with torch.no_grad():
        avg_speed = speed_history.mean(0)
        success = torch.all(distance.flatten(0, 1) > 0, 0)
        _success = success.sum() / B
        smooth_dict({
            'loss': loss,
            'loss_v': loss_v,
            'loss_v_pred': loss_v_pred,
            'loss_obj_avoidance': loss_obj_avoidance,
            'loss_d_acc': loss_d_acc,
            'loss_d_jerk': loss_d_jerk,
            'loss_d_snap': loss_d_snap,
            'loss_bias': loss_bias,
            'loss_speed': loss_speed,
            'loss_collide': loss_collide,
            'loss_ground_affinity': loss_ground_affinity,
            'success': _success,
            'max_speed': speed_history.max(0).values.mean(),
            'avg_speed': avg_speed.mean(),
            'ar': (success * avg_speed).mean()})
        log_dict = {}
        if is_save_iter(i):
            # vid = torch.stack(vid).cpu().div(10).clamp(0, 1)[None, :, None]
            fig_p, ax = plt.subplots()
            p_history = p_history[:, 4].cpu()
            ax.plot(p_history[:, 0], label='x')
            ax.plot(p_history[:, 1], label='y')
            ax.plot(p_history[:, 2], label='z')
            ax.legend()
            fig_v, ax = plt.subplots()
            v_history = v_history[:, 4].cpu()
            ax.plot(v_history[:, 0], label='x')
            ax.plot(v_history[:, 1], label='y')
            ax.plot(v_history[:, 2], label='z')
            ax.legend()
            fig_a, ax = plt.subplots()
            act_buffer = act_buffer[:, 4].cpu()
            ax.plot(act_buffer[:, 0], label='x')
            ax.plot(act_buffer[:, 1], label='y')
            ax.plot(act_buffer[:, 2], label='z')
            ax.legend()
            # writer.add_video('demo', vid, i + 1, 15)
            writer.add_figure('p_history', fig_p, i + 1)
            writer.add_figure('v_history', fig_v, i + 1)
            writer.add_figure('a_reals', fig_a, i + 1)
        if (i + 1) % 10000 == 0:
            torch.save(model.state_dict(), f'{args.ckpt_dir}/checkpoint{i//10000:04d}.pth')
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            scaler_q.clear()
