import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

def _save_img01(img01, path):
    arr = (img01.clamp(0,1).detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

@torch.no_grad()
def lift_points_and_colors(rgb01, depth, K, stride=4, max_points=200000, depth_min=1e-4):
    # rgb01: (3,H,W) in [0,1], depth:(H,W)
    device = depth.device
    H, W = depth.shape
    ys = torch.arange(0, H, stride, device=device)
    xs = torch.arange(0, W, stride, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    d = depth[yy, xx]
    valid = torch.isfinite(d) & (d > depth_min)
    yy = yy[valid].float()
    xx = xx[valid].float()
    d  = d[valid].float()

    if max_points is not None and d.numel() > max_points:
        idx = torch.randperm(d.numel(), device=device)[:max_points]
        yy, xx, d = yy[idx], xx[idx], d[idx]

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    X = (xx - cx) / fx * d
    Y = (yy - cy) / fy * d
    Z = d
    pts = torch.stack([X,Y,Z], dim=-1)  # (N,3)

    xi = xx.round().long().clamp(0, W-1)
    yi = yy.round().long().clamp(0, H-1)
    cols = rgb01[:, yi, xi].permute(1,0).contiguous()  # (N,3)
    return pts, cols

@torch.no_grad()
def apply_rel_w2c(pts_c1, T1_w2c, T2_w2c):
    # pts_c1: (N,3)
    N = pts_c1.shape[0]
    ones = torch.ones((N,1), device=pts_c1.device, dtype=pts_c1.dtype)
    pts_h = torch.cat([pts_c1, ones], dim=1)  # (N,4)
    T_rel = T2_w2c @ torch.linalg.inv(T1_w2c)  # exactly same as Warper
    pts_c2 = (pts_h @ T_rel.T)[:, :3]
    return pts_c2

@torch.no_grad()
def render_zbuffer(pts_c2, cols, K, H, W, z_min=1e-4):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    x,y,z = pts_c2[:,0], pts_c2[:,1], pts_c2[:,2]
    valid = torch.isfinite(z) & (z > z_min)
    x,y,z,cols = x[valid], y[valid], z[valid], cols[valid]

    u = (fx*(x/z) + cx).round().long()
    v = (fy*(y/z) + cy).round().long()
    inb = (u>=0)&(u<W)&(v>=0)&(v<H)
    u,v,z,cols = u[inb], v[inb], z[inb], cols[inb]

    idx = v*W + u
    order = torch.argsort(z)  # near first
    idx, cols = idx[order], cols[order]
    order2 = torch.argsort(idx, stable=True)
    idx, cols = idx[order2], cols[order2]

    keep = torch.ones_like(idx, dtype=torch.bool)
    keep[1:] = idx[1:] != idx[:-1]
    idx, cols = idx[keep], cols[keep]

    img = torch.zeros((H*W,3), device=cols.device, dtype=cols.dtype)
    mask = torch.zeros((H*W,), device=cols.device, dtype=torch.bool)
    img[idx] = cols
    mask[idx] = True
    return img.view(H,W,3), mask.view(H,W)

@torch.no_grad()
def camera_centers_from_w2c(T_w2c):  # (T,4,4)
    R = T_w2c[:, :3, :3]
    t = T_w2c[:, :3, 3:4]
    C = (-R.transpose(1,2) @ t)[:, :, 0]
    return C  # (T,3)

@torch.no_grad()
def debug_viz_pc_and_pose(frames, depths, pose_s_w2c, pose_t_w2c, K, save_dir, idx=0):
    os.makedirs(save_dir, exist_ok=True)

    rgb01 = (frames[idx].clamp(-1,1) + 1) / 2.0     # (3,H,W)
    depth = depths[idx,0]                           # (H,W)
    Ki = K[idx]
    H, W = depth.shape

    pts_c1, cols = lift_points_and_colors(rgb01, depth, Ki, stride=4)
    pts_c2 = apply_rel_w2c(pts_c1, pose_s_w2c[idx], pose_t_w2c[idx])
    img_r, mask_r = render_zbuffer(pts_c2, cols, Ki, H, W)

    _save_img01(img_r, os.path.join(save_dir, f"render_{idx:03d}.png"))
    _save_img01(mask_r.float().unsqueeze(-1).repeat(1,1,3), os.path.join(save_dir, f"mask_{idx:03d}.png"))

    # 画 “点云(用source cam坐标) + target camera trajectory(世界坐标)”
    # world = inv(w2c) * cam
    ones = torch.ones((pts_c1.shape[0],1), device=pts_c1.device, dtype=pts_c1.dtype)
    pts_h = torch.cat([pts_c1, ones], dim=1)
    T_c2w = torch.linalg.inv(pose_s_w2c[idx])
    pts_w = (pts_h @ T_c2w.T)[:, :3]

    C_traj = camera_centers_from_w2c(pose_t_w2c)  # (T,3)

    # downsample for plotting
    maxp = 60000
    if pts_w.shape[0] > maxp:
        sel = torch.randperm(pts_w.shape[0], device=pts_w.device)[:maxp]
        pts_w = pts_w[sel]
        cols_p = cols[sel]
    else:
        cols_p = cols

    p = pts_w.detach().cpu().numpy()
    c = cols_p.detach().cpu().numpy()
    C = C_traj.detach().cpu().numpy()

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(p[:,0], p[:,1], p[:,2], s=0.2, c=c)
    ax.plot(C[:,0], C[:,1], C[:,2], linewidth=2)
    ax.scatter(C[0,0], C[0,1], C[0,2], s=50)
    ax.set_title("Lifted 3D points + target camera trajectory")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"pc_traj_{idx:03d}.png"), dpi=200)
    plt.close(fig)
