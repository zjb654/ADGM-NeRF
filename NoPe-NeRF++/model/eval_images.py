import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import numpy as np
import cv2
import imageio
from model.common import mse2psnr
from third_party import pytorch_ssim
from skimage import metrics
from model.common import (
    get_tensor_values, arange_pixels
)

logger_py = logging.getLogger(__name__)


class Eval_Images(object):

    def __init__(self, renderer, cfg, points_batch_size=100000, use_learnt_poses=True,
                 use_learnt_focal=True, device=None, render_type=None, c2ws=None, img_list=None):
        self.points_batch_size = points_batch_size
        self.renderer = renderer

        # ===== Diagnostic 1: record config values =====
        self.resolution = cfg['extract_images']['resolution']
        print(f"\n[DIAGNOSTIC] Eval_Images initialized:")
        print(f"  - cfg['extract_images']['resolution'] = {self.resolution}")
        print(
            f"  - type: {type(self.resolution)}, length: {len(self.resolution) if hasattr(self.resolution, '__len__') else 'N/A'}")

        # Config validation
        if not isinstance(self.resolution, (list, tuple)):
            raise TypeError(f"Config error: resolution must be list/tuple, got {type(self.resolution)}: {self.resolution}")
        if len(self.resolution) != 2:
            raise ValueError(
                f"Config error: resolution must contain 2 elements [H,W], got {len(self.resolution)}: {self.resolution}")

        self.device = device
        self.use_learnt_poses = use_learnt_poses
        self.use_learnt_focal = use_learnt_focal
        self.render_type = render_type
        self.c2ws = c2ws
        self.img_list = img_list

    def process_data_dict(self, data):
        '''Process the data dictionary and return tensors'''
        device = self.device
        img = data.get('img').to(device)
        batch_size, _, h, w = img.shape
        depth_img = data.get('img.depth', torch.ones(batch_size, h, w))
        img_idx = data.get('img.idx')
        camera_mat = data.get('img.camera_mat').to(device)
        scale_mat = data.get('img.scale_mat').to(device)

        # ===== Diagnostic 2: actual GT image size =====
        print(f"\n[DIAGNOSTIC] process_data_dict:")
        print(f"  - Loaded GT image size: {img.shape} (Batch, C, H, W)")
        print(f"  - Actual H={h}, W={w}")

        return (img, depth_img, camera_mat, scale_mat, img_idx)

    def eval_images(self, data, render_dir, fxfy, lpips_vgg_fn, logger,
                    min_depth=0.1, max_depth=20, it=0):
        self.renderer.eval()
        (img_gt, depth_gt, camera_mat, scale_mat, img_idx) = self.process_data_dict(data)
        img_idx = int(img_idx)

        # Convert GT to (H,W,3) format
        img_gt = img_gt.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
        gt_h, gt_w = img_gt.shape[0], img_gt.shape[1]

        # ===== Diagnostic 3: config resolution vs actual GT size =====
        cfg_h, cfg_w = self.resolution
        print(f"\n[DIAGNOSTIC] Size comparison:")
        print(f"  - Resolution from config: ({cfg_h}, {cfg_w})")
        print(f"  - Actual loaded GT size: ({gt_h}, {gt_w})")

        # Intelligently handle size mismatch
        if (cfg_h, cfg_w) != (gt_h, gt_w):
            print(f"  [WARNING] Size mismatch! config({cfg_h},{cfg_w}) != GT({gt_h},{gt_w})")
            print(f"  Will render at GT size ({gt_h},{gt_w}) to avoid errors")
            h, w = gt_h, gt_w  # Force using GT size
        else:
            h, w = cfg_h, cfg_w
            print(f"  Size match, using config resolution")

        depth_gt = depth_gt.squeeze(0).numpy()
        mask = (depth_gt > min_depth) * (depth_gt < max_depth)

        if self.use_learnt_poses:
            c2w = self.c2ws[img_idx]
            world_mat = torch.inverse(c2w).unsqueeze(0)

        if self.use_learnt_focal:
            camera_mat = torch.tensor([[[fxfy[0], 0, 0, 0],
                                        [0, -fxfy[1], 0, 0],
                                        [0, 0, -1, 0],
                                        [0, 0, 0, 1]]]).to(self.device)

        # ===== Diagnostic 4: resolution for rendering =====
        print(f"\n[DIAGNOSTIC] Starting rendering:")
        print(f"  - Target resolution: ({h}, {w})")

        p_loc, pixels = arange_pixels(resolution=(h, w))
        pixels = pixels.to(self.device)

        # Prepare depth input (dummy)
        depth_input = torch.zeros(1, 1, h, w).to(self.device)
        depth_input = get_tensor_values(depth_input, pixels.clone(), mode='nearest', scale=True, detach=False)
        depth_input = torch.ones_like(depth_input)

        # Render
        with torch.no_grad():
            rgb_pred = []
            depth_pred = []
            for i, (pixels_i, depth_i) in enumerate(zip(
                    torch.split(pixels, self.points_batch_size, dim=1),
                    torch.split(depth_input, self.points_batch_size, dim=1)
            )):
                out_dict = self.renderer(pixels_i, depth_i, camera_mat, world_mat, scale_mat,
                                         self.render_type, eval_=True, it=it, add_noise=False)
                rgb_pred_i = out_dict['rgb']
                depth_pred_i = out_dict['depth_pred']
                rgb_pred.append(rgb_pred_i)
                depth_pred.append(depth_pred_i)

            rgb_pred = torch.cat(rgb_pred, dim=1)
            img_out = rgb_pred.view(h, w, 3)
            depth_pred = torch.cat(depth_pred, dim=0)
            depth_pred = depth_pred.view(h, w).detach().cpu().numpy()

        # ===== Diagnostic 5: rendered output size =====
        print(f"\n[DIAGNOSTIC] Rendering done:")
        print(f"  - Rendered output size: {img_out.shape}")
        print(f"  - GT image size:        {img_gt.shape}")

        # Final size alignment check (defensive programming)
        if img_out.shape[0:2] != img_gt.shape[0:2]:
            print(f"  [WARNING] Final size still mismatched! rendered{img_out.shape[0:2]} vs GT{img_gt.shape[0:2]}")
            print(f"  [FIX] Resizing rendered result to GT size...")

            # Save the original high-resolution rendering (for visualization)
            img_out_hr = img_out.clone()

            # Resize rendered output to GT size (downsample)
            img_out = F.interpolate(
                img_out.permute(2, 0, 1).unsqueeze(0),
                size=(img_gt.shape[0], img_gt.shape[1]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)

            # Depth map must be resized too (nearest neighbor to avoid blurring)
            depth_pred = cv2.resize(depth_pred, (img_gt.shape[1], img_gt.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            print(f"  [FIX] Resize done: img_out={img_out.shape}, depth_pred={depth_pred.shape}")

        # Assert sizes match (prevent downstream errors)
        assert img_out.shape == img_gt.shape, \
            f"[ERROR] Size alignment failed: rendered{img_out.shape} vs GT{img_gt.shape}"

        print(f"  Starting metric computation (MSE/PSNR/SSIM/LPIPS)...")

        # Compute MSE and PSNR
        mse = F.mse_loss(img_out, img_gt).item()
        psnr = mse2psnr(mse)

        # SSIM computation (requires B,C,H,W format)
        img_out_bcwh = img_out.permute(2, 0, 1).unsqueeze(0)
        img_gt_bcwh = img_gt.permute(2, 0, 1).unsqueeze(0)
        ssim = pytorch_ssim.ssim(img_out_bcwh, img_gt_bcwh).item()

        # LPIPS
        lpips_loss = lpips_vgg_fn(img_out_bcwh.contiguous(),
                                  img_gt_bcwh.contiguous(),
                                  normalize=True).item()

        tqdm.write(f'[Image {img_idx:04d}] PSNR: {psnr:.2f}, SSIM: {ssim:.2f}, LPIPS: {lpips_loss:.2f}')

        # Create output directory
        gt_height, gt_width = img_gt.shape[0:2]

        img_out_dir = os.path.join(render_dir, 'img_out')
        depth_out_dir = os.path.join(render_dir, 'depth_out')
        img_gt_dir = os.path.join(render_dir, 'img_gt_out')

        for d in [img_out_dir, depth_out_dir, img_gt_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        # Save images (as uint8)
        depth_min, depth_max = depth_pred.min(), depth_pred.max()
        if depth_max > depth_min:
            depth_out_vis = (np.clip(255.0 / (depth_max - depth_min) * (depth_pred - depth_min), 0, 255)).astype(
                np.uint8)
        else:
            depth_out_vis = np.zeros_like(depth_pred, dtype=np.uint8)

        img_out_uint8 = (img_out.cpu().numpy() * 255).astype(np.uint8)
        img_gt_uint8 = (img_gt.cpu().numpy() * 255).astype(np.uint8)

        imageio.imwrite(os.path.join(img_out_dir, str(img_idx).zfill(4) + '.png'), img_out_uint8)
        imageio.imwrite(os.path.join(depth_out_dir, str(img_idx).zfill(4) + '.png'), depth_out_vis)
        imageio.imwrite(os.path.join(img_gt_dir, str(img_idx).zfill(4) + '.png'), img_gt_uint8)

        # Valid region for depth evaluation
        depth_out_eval = depth_pred[mask]
        depth_gt_eval = depth_gt[mask]

        return {
            'img': img_out_uint8,
            'depth': depth_out_vis,
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips_loss,
            'depth_pred': depth_out_eval,
            'depth_gt': depth_gt_eval
        }