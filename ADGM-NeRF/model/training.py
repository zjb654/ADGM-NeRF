import os
import torch
import logging
from model.losses import Loss
import numpy as np
from PIL import Image
import imageio
from torch.nn import functional as F
from model.common import (
    get_tensor_values,
    arange_pixels, project_to_cam, transform_to_world,
)
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

logger_py = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, model, optimizer, cfg, device=None, optimizer_pose=None, pose_param_net=None,
                 optimizer_focal=None, focal_net=None, optimizer_distortion=None, distortion_net=None,
                 train_dataset=None, train_loader=None, **kwargs):
        """model trainer
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.optimizer_pose = optimizer_pose
        self.pose_param_net = pose_param_net
        self.focal_net = focal_net
        self.optimizer_focal = optimizer_focal
        self.distortion_net = distortion_net
        self.optimizer_distortion = optimizer_distortion

        self.train_dataset = train_dataset
        self.train_loader = train_loader

        self.n_training_points = cfg['n_training_points']
        self.rendering_technique = cfg['type']
        self.vis_geo = cfg['vis_geo']

        self.detach_gt_depth = cfg['detach_gt_depth']
        self.pc_ratio = cfg['pc_ratio']
        self.match_method = cfg['match_method']
        self.shift_first = cfg['shift_first']
        self.detach_ref_img = cfg['detach_ref_img']
        self.scale_pcs = cfg['scale_pcs']
        self.detach_rgbs_scale = cfg['detach_rgbs_scale']
        self.vis_reprojection_every = cfg['vis_reprojection_every']
        self.nearest_limit = cfg['nearest_limit']
        self.annealing_epochs = cfg['annealing_epochs']

        self.rgb_weight_cfg = cfg['rgb_weight']
        self.depth_weight_cfg = cfg['depth_weight']

        self.pc_weight = cfg['pc_weight']
        self.rgb_s_weight = cfg['rgb_s_weight']
        self.rgb_weight = cfg['rgb_weight']
        self.depth_weight = cfg['depth_weight']
        self.weight_dist_2nd_loss = cfg['weight_dist_2nd_loss']
        self.weight_dist_1st_loss = cfg['weight_dist_1st_loss']
        self.depth_consistency_weight = cfg['depth_consistency_weight']

        # ---- EGGM (explicit global geometric matching, paper-faithful) ----
        # iterations at which the peak-anchor maps are refreshed (5w/10w/15w)
        self.eggm_update_iters = cfg.get('eggm_update_iters', [50000, 100000, 150000])
        # global loss is active from the first refresh onwards
        self.eggm_start_it = cfg.get('eggm_start_it', 50000)
        # linear warm-up of the global weight: 0 -> eggm_warmup_weight
        # over eggm_warmup_iters iterations after activation
        self.eggm_warmup_iters = cfg.get('eggm_warmup_iters', 5000)
        self.eggm_warmup_weight = cfg.get('eggm_warmup_weight', 0.1)
        # view-pair filters (identical to the paper: |i-j| >= 2, tau_t < 90 deg)
        self.eggm_min_pair_gap = cfg.get('eggm_min_pair_gap', 2)
        self.eggm_max_view_angle = cfg.get('eggm_max_view_angle', 90.0)
        # match acceptance: 3D distance < tau_x = 0.02 * scene diagonal
        self.eggm_dist_thresh_rel = cfg.get('eggm_dist_thresh_rel', 0.02)
        # max cached anchor points per view (uniform subsample, deterministic)
        self.eggm_max_points_per_view = cfg.get('eggm_max_points_per_view', 20000)
        # number of reference keyframes matched per training step
        self.eggm_refs_per_step = cfg.get('eggm_refs_per_step', 1)
        self._eggm_cache = None          # {view_idx: {'pts': (M,3) cpu, 'pix': (M,2) cpu}}
        self._eggm_cache_it = -1         # iteration of the last refresh
        self._eggm_scene_diag = None     # scene diagonal at cache time

        self.loss = Loss(cfg)

    def train_step(self, data, it=None, epoch=None, scheduling_start=None, render_path=None):
        self.model.train()
        self.optimizer.zero_grad()
        if self.pose_param_net:
            self.pose_param_net.train()
            self.optimizer_pose.zero_grad()
        if self.focal_net:
            self.focal_net.train()
            self.optimizer_focal.zero_grad()
        if self.distortion_net:
            self.distortion_net.train()
            self.optimizer_distortion.zero_grad()

        if self.focal_net:
            fxfy = self.focal_net(0)
        else:
            fxfy = None

        T = None

        if self.distortion_net:
            scale_input, shift_input = self.distortion_net(data['img.idx'])
        else:
            scale_input, shift_input = None, None

        loss_dict = self.compute_loss(data, it=it, epoch=epoch, scheduling_start=scheduling_start,
                                      out_render_path=render_path, fxfy=fxfy, T=T)
        loss = loss_dict['loss']
        loss.backward()
        self.optimizer.step()
        if self.optimizer_pose:
            self.optimizer_pose.step()
        if self.optimizer_focal:
            self.optimizer_focal.step()
        if self.optimizer_distortion:
            self.optimizer_distortion.step()
        return loss_dict

    def render_visdata(self, data, resolution, it, out_render_path):
        (img, dpt, camera_mat, scale_mat, img_idx) = self.process_data_dict(data)
        h, w = resolution
        if self.pose_param_net:
            c2w = self.pose_param_net(img_idx)
            world_mat = torch.inverse(c2w).unsqueeze(0)
        if self.optimizer_focal:
            fxfy = self.focal_net(0)
            camera_mat = torch.tensor([[[fxfy[0], 0, 0, 0],
                                        [0, -fxfy[1], 0, 0],
                                        [0, 0, -1, 0],
                                        [0, 0, 0, 1]]]).to(self.device)
        p_idx = torch.arange(h * w).to(self.device)
        p_loc, pixels = arange_pixels(resolution=(h, w))

        pixels = pixels.to(self.device)
        depth_input = dpt

        with torch.no_grad():
            rgb_pred = []
            depth_pred = []
            for i, (pixels_i, p_idx_i) in enumerate(
                    zip(torch.split(pixels, 1024, dim=1), torch.split(p_idx, 1024, dim=0))):
                out_dict = self.model(
                    pixels_i, p_idx_i, camera_mat, world_mat, scale_mat, self.rendering_technique,
                    add_noise=False, eval_mode=True, it=it, depth_img=depth_input, img_size=(h, w))
                rgb_pred_i = out_dict['rgb']
                rgb_pred.append(rgb_pred_i)
                depth_pred_i = out_dict['depth_pred']
                depth_pred.append(depth_pred_i)

            rgb_pred = torch.cat(rgb_pred, dim=1)
            depth_pred = torch.cat(depth_pred, dim=0)

            rgb_pred = rgb_pred.view(h, w, 3).detach().cpu().numpy()
            img_out = (rgb_pred * 255).astype(np.uint8)

            img1 = Image.fromarray(
                (img_out).astype(np.uint8)
            ).convert("RGB").save(
                os.path.join(out_render_path, '%04d_img.png' % img_idx)
            )
        if self.vis_geo:
            with torch.no_grad():
                rgb_pred = \
                    [self.model(
                        pixels_i, None, camera_mat, world_mat, scale_mat, 'phong_renderer',
                        add_noise=False, eval_mode=True, it=it, depth_img=depth_input, img_size=(h, w))['rgb']
                     for i, pixels_i in enumerate(torch.split(pixels, 1024, dim=1))]

                rgb_pred = torch.cat(rgb_pred, dim=1).cpu()
                rgb_pred = rgb_pred.view(h, w, 3).detach().cpu().numpy()
                img_out = (rgb_pred * 255).astype(np.uint8)

                img1 = Image.fromarray(
                    (img_out).astype(np.uint8)
                ).convert("RGB").save(
                    os.path.join(out_render_path, '%04d_geo.png' % img_idx)
                )

        return img_out.astype(np.uint8)

    def process_data_dict(self, data):
        device = self.device
        img = data.get('img').to(device)
        img_idx = data.get('img.idx')
        dpt = data.get('img.dpt').to(device).unsqueeze(1)
        camera_mat = data.get('img.camera_mat').to(device)
        scale_mat = data.get('img.scale_mat').to(device)
        return (img, dpt, camera_mat, scale_mat, img_idx)

    def process_data_reference(self, data):
        device = self.device
        ref_imgs = data.get('img.ref_imgs').to(device)
        ref_dpts = data.get('img.ref_dpts').to(device).unsqueeze(1)
        ref_idxs = data.get('img.ref_idxs')
        return (ref_imgs, ref_dpts, ref_idxs)

    def anneal(self, start_weight, end_weight, anneal_start_epoch, anneal_epoches, current):
        if current <= anneal_start_epoch:
            return start_weight
        elif current >= anneal_start_epoch + anneal_epoches:
            return end_weight
        else:
            return start_weight + (end_weight - start_weight) * (current - anneal_start_epoch) / anneal_epoches

    def get_sigmoid_alpha_by_iter(self, it, end_it=100000, steepness=6.0):
        if it is None or it <= 0:
            return 0.5
        current_it = min(it, end_it)
        progress = current_it / float(end_it)
        x = progress * steepness
        alpha = 1.0 / (1.0 + np.exp(-x))
        return alpha

    def get_peak_depth(self, weights, z_vals, threshold=0):
        """Extract the depth of maximum weight (peak depth)."""
        max_w_val, max_w_idx = torch.max(weights, dim=-1)
        peak_depth = torch.gather(z_vals, -1, max_w_idx.unsqueeze(-1)).squeeze(-1)
        valid_mask = max_w_val > threshold
        return peak_depth, valid_mask

    @torch.no_grad()
    def update_eggm_anchor_cache(self, it):
        ''' Refreshes the global peak-anchor maps with the CURRENT model and
        poses (paper: matching points are updated every 50k iterations).

        For every training view the peak-depth map is rendered (no grad),
        unprojected to world-space anchor points with the current pose, and
        cached on CPU together with the source pixel coordinates. Cached
        anchors are constants w.r.t. the graph (stop-gradient by design).
        Also recomputes the scene diagonal used for the tau_x threshold. '''
        if self.train_dataset is None or self.pose_param_net is None:
            return
        self.model.eval()
        self.pose_param_net.eval()
        device = self.device
        n_views = self.pose_param_net.num_cams
        # train_dataset is a dict of DataFields (not indexable); the only
        # reliable per-view access is the DataLoader. Rebuild a non-shuffled
        # loader over the same dataset/collate so the live training iterator
        # is not disturbed.
        if self.train_loader is None:
            raise RuntimeError(
                'EGGM anchor cache needs train_loader; pass '
                'train_loader=train_loader when constructing the Trainer.')
        from torch.utils.data import DataLoader
        loader = DataLoader(self.train_loader.dataset, batch_size=1,
                            shuffle=False,
                            collate_fn=self.train_loader.collate_fn)
        cache = {}
        for data in loader:
            idx_t = data.get('img.idx')
            idx = int(idx_t.item()) if torch.is_tensor(idx_t) else int(idx_t)
            h, w = data['img'].shape[-2:]
            info = self.extract_dual_depths(data, (h, w), it)
            valid = info['valid_peak'].view(-1).to(device)
            if valid.sum() == 0:
                continue
            depth = info['depth_peak'].view(-1).to(device)[valid]
            _, pix = arange_pixels(resolution=(h, w), device=device)
            pix_v = pix[0, valid]                                  # (M, 2)
            cam = info['camera_mat'].to(device)
            c2w = info['c2w'].to(device).float()
            if c2w.dim() == 2:
                c2w = c2w.unsqueeze(0)
            P_cam = transform_to_world(pix_v.unsqueeze(0),
                                       depth.view(1, -1, 1), cam)
            P_world = (P_cam @ c2w[:, :3, :3].transpose(1, 2)
                       + c2w[:, :3, 3]).squeeze(0)                # (M, 3)
            # deterministic uniform subsample (same indices every refresh)
            max_pts = self.eggm_max_points_per_view
            if max_pts and P_world.shape[0] > max_pts:
                sel = torch.randperm(0, P_world.shape[0] - 1, max_pts,
                                     device=device).long()
                P_world = P_world[sel]
                pix_v = pix_v[sel]
            cache[idx] = {'pts': P_world.detach().cpu(),
                          'pix': pix_v.detach().cpu()}
        if len(cache) < 2:
            logger_py.warning('[EGGM] anchor cache too small, disabled.')
            self._eggm_cache = None
            self.model.train()
            self.pose_param_net.train()
            return
        all_pts = torch.cat([c['pts'] for c in cache.values()], dim=0)
        lo = torch.quantile(all_pts, 0.01, dim=0)
        hi = torch.quantile(all_pts, 0.99, dim=0)
        self._eggm_scene_diag = float((hi - lo).norm().item())
        self._eggm_cache = cache
        self._eggm_cache_it = it
        self.model.train()
        self.pose_param_net.train()
        logger_py.info('[EGGM] anchor cache refreshed @it=%d: %d views, '
                       'scene diagonal %.4f', it, len(cache),
                       self._eggm_scene_diag)

    def compute_global_loss(self, img_idx, out_dict_curr, p_curr, camera_mat, device):
        ''' Explicit global geometric matching loss (paper Eq. 11).

        Current-frame peak points (LIVE gradients) are unprojected with the
        current pose, reprojected into a historical keyframe j
        (|i-j| >= min gap, viewing angle < tau_t), matched to the cached
        DETACHED anchor points of j by nearest neighbour in image space, and
        accepted if the 3D distance is below tau_x = 2% of the scene
        diagonal. The loss is the mean L1 distance over accepted pairs.
        Gradients flow into the current depth anchor and BOTH poses, but not
        into the cached target anchors. '''
        zero = torch.zeros((), device=device)
        if self.pose_param_net is None or self._eggm_cache is None:
            return zero
        curr_idx = int(img_idx.item()) if torch.is_tensor(img_idx) else int(img_idx)
        cache = self._eggm_cache

        # ---- current-frame peak anchor (live) ----
        if 'weights' in out_dict_curr and 'z_vals' in out_dict_curr:
            weights = out_dict_curr['weights']
            z_vals = out_dict_curr['z_vals']
            if weights.dim() == 3:
                weights = weights.squeeze(0)
                z_vals = z_vals.squeeze(0)
            peak_depth, valid_mask = self.get_peak_depth(weights, z_vals, threshold=0)
            anchor_depth = peak_depth.view(1, -1, 1)
        else:
            anchor_depth = out_dict_curr['depth_pred'].view(1, -1, 1)
            valid_mask = torch.ones(anchor_depth.shape[1], dtype=torch.bool,
                                    device=device)
        if valid_mask.sum() == 0:
            return zero

        c2w_curr = self.pose_param_net(img_idx)
        if c2w_curr.dim() == 2:
            c2w_curr = c2w_curr.unsqueeze(0)
        P_cam = transform_to_world(p_curr[:, valid_mask, :],
                                   anchor_depth[:, valid_mask, :], camera_mat)
        P_world = (P_cam @ c2w_curr[:, :3, :3].transpose(1, 2)
                   + c2w_curr[:, :3, 3]).squeeze(0)               # (N, 3) live

        # ---- keyframe selection: |i-j| >= gap and viewing angle < tau_t ----
        import math
        idxs = sorted(cache.keys())
        cand = [j for j in idxs
                if abs(j - curr_idx) >= self.eggm_min_pair_gap]
        if len(cand) == 0:
            return zero
        # viewing directions from CURRENT poses (camera forward = -z axis)
        fwd = {}
        with torch.no_grad():
            all_idx = [curr_idx] + cand
            c2ws = torch.stack(
                [self.pose_param_net(torch.tensor(vi, device=device))
                 for vi in all_idx],
                dim=0)                              # (K, 4, 4)
            R = c2ws[:, :3, :3]
            f = R @ torch.tensor([0., 0., -1.], device=device).view(1, 3, 1)
            f = (f / f.norm(dim=1, keepdim=True)).squeeze(-1)     # (K, 3)
            for k, vi in enumerate(all_idx):
                fwd[vi] = f[k]
        cos_min = math.cos(math.radians(self.eggm_max_view_angle))
        cand = [j for j in cand
                if float((fwd[curr_idx] * fwd[j]).sum()) > cos_min]
        if len(cand) == 0:
            return zero
        n_refs = min(self.eggm_refs_per_step, len(cand))
        refs = list(np.random.choice(cand, size=n_refs, replace=False))

        thresh = self.eggm_dist_thresh_rel * self._eggm_scene_diag
        losses = []
        for j in refs:
            c2w_ref = self.pose_param_net(
                torch.tensor([j], device=device))                 # live pose
            if c2w_ref.dim() == 2:
                c2w_ref = c2w_ref.unsqueeze(0)
            w2c_ref = torch.inverse(c2w_ref)
            P_cam_ref = (P_world @ w2c_ref[:, :3, :3].transpose(1, 2)
                         + w2c_ref[:, :3, 3])
            p_2d, valid_proj = project_to_cam(P_cam_ref, camera_mat, device)
            valid_proj = valid_proj.bool().squeeze()
            if valid_proj.sum() == 0:
                continue
            P_src = P_world[valid_proj]                          # (M, 3) live
            q = p_2d[0, valid_proj, :]                           # (M, 2)
            # nearest neighbour in the keyframe image plane, then keep the
            # single best (minimum-distance) target per source point
            pix_ref = cache[j]['pix'].to(device)                 # (K, 2) const
            pts_ref = cache[j]['pts'].to(device)                 # (K, 3) const
            nn_idx = torch.zeros(q.shape[0], dtype=torch.long, device=device)
            for s in range(0, q.shape[0], 4096):
                nn_idx[s:s + 4096] = torch.cdist(
                    q[s:s + 4096].unsqueeze(0),
                    pix_ref.unsqueeze(0)).squeeze(0).argmin(dim=1)
            P_tgt = pts_ref[nn_idx]                              # detached
            dist = (P_src - P_tgt).norm(dim=-1)
            accept = dist < thresh
            if accept.any():
                losses.append(torch.abs(P_src[accept] - P_tgt[accept]).mean())
        if len(losses) == 0:
            return zero
        return torch.stack(losses).mean()

    @torch.no_grad()
    def extract_dual_depths(self, data, resolution, it, fxfy=None, out_dir=None):
        """
        Extract two depth maps for a single frame (both are camera-space Z depths):
        - depth_expected: ray-integration (expected) depth
        - depth_peak:     maximum-opacity depth (converted to Z depth via ray_vector_norm)
        Only .png visualizations are saved; .npy files are not saved.
        """
        self.model.eval()
        if self.pose_param_net:
            self.pose_param_net.eval()

        (img, dpt, camera_mat_gt, scale_mat, img_idx) = self.process_data_dict(data)
        h, w = resolution

        if self.pose_param_net:
            c2w = self.pose_param_net(img_idx)
            world_mat = torch.inverse(c2w).unsqueeze(0)
        else:
            world_mat = data.get('img.world_mat').to(self.device).unsqueeze(0)
            c2w = torch.inverse(world_mat.squeeze(0))

        if self.optimizer_focal:
            if fxfy is None:
                fxfy = self.focal_net(0)
            camera_mat = torch.tensor([[[fxfy[0], 0, 0, 0],
                                        [0, -fxfy[1], 0, 0],
                                        [0, 0, -1, 0],
                                        [0, 0, 0, 1]]]).to(self.device)
        else:
            camera_mat = camera_mat_gt

        p_loc, pixels = arange_pixels(resolution=(h, w))
        pixels = pixels.to(self.device)
        p_idx = torch.arange(h * w).to(self.device)

        depth_input = torch.ones(1, 1, h, w).to(self.device)

        depth_expected_list = []
        depth_peak_list = []
        valid_peak_list = []
        rvn_list = []

        split_size = 1024
        for pixels_i, p_idx_i in zip(
                torch.split(pixels, split_size, dim=1),
                torch.split(p_idx, split_size, dim=0)):
            out_dict = self.model(
                pixels_i, p_idx_i, camera_mat, world_mat, scale_mat,
                self.rendering_technique,
                add_noise=False, eval_mode=True, it=it,
                depth_img=depth_input, img_size=(h, w)
            )

            if 'depth_pred_full' in out_dict:
                depth_expected_list.append(out_dict['depth_pred_full'])
            else:
                depth_expected_list.append(out_dict['depth_pred'])

            if 'weights' in out_dict and 'z_vals' in out_dict:
                weights = out_dict['weights']
                z_vals = out_dict['z_vals']
                if weights.dim() == 3 and weights.shape[0] == 1:
                    weights = weights.squeeze(0)
                if z_vals.dim() == 3 and z_vals.shape[0] == 1:
                    z_vals = z_vals.squeeze(0)
                elif z_vals.dim() == 2:
                    pass
                peak_depth, valid_peak = self.get_peak_depth(weights, z_vals, threshold=0)
                depth_peak_list.append(peak_depth)
                valid_peak_list.append(valid_peak)
            else:
                dp = out_dict['depth_pred']
                depth_peak_list.append(dp)
                valid_peak_list.append(torch.ones_like(dp, dtype=torch.bool))

            if 'ray_vector_norm' in out_dict:
                rvn_list.append(out_dict['ray_vector_norm'])

        depth_expected = torch.cat(depth_expected_list, dim=0)
        depth_peak = torch.cat(depth_peak_list, dim=0)
        valid_peak = torch.cat(valid_peak_list, dim=0)

        if depth_expected.numel() == h * w:
            depth_expected = depth_expected.view(h, w)
        if depth_peak.numel() == h * w:
            depth_peak = depth_peak.view(h, w)
        if valid_peak.numel() == h * w:
            valid_peak = valid_peak.view(h, w)

        if len(rvn_list) > 0:
            ray_vector_norm_full = torch.cat(rvn_list, dim=0)
            if ray_vector_norm_full.numel() == h * w:
                ray_vector_norm_full = ray_vector_norm_full.view(h, w)
                depth_peak = depth_peak / (ray_vector_norm_full + 1e-8)

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            idx = img_idx.item() if torch.is_tensor(img_idx) else img_idx
            cmap = plt.cm.turbo

            dep_exp_np = depth_expected.cpu().numpy()
            mask_valid = dep_exp_np > 1e-6
            if mask_valid.any():
                vmin, vmax = dep_exp_np[mask_valid].min(), dep_exp_np[mask_valid].max()
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                dep_color = np.ones((h, w, 3), dtype=np.uint8) * 255
                dep_color[mask_valid] = (cmap(norm(dep_exp_np[mask_valid]))[:, :3] * 255).astype(np.uint8)
            else:
                dep_color = np.zeros((h, w, 3), dtype=np.uint8)
            imageio.imwrite(os.path.join(out_dir, f'{idx:04d}_depth_expected.png'), dep_color)

            dep_peak_np = depth_peak.cpu().numpy()
            mask_peak_valid = valid_peak.cpu().numpy() & (dep_peak_np > 1e-6)
            if mask_peak_valid.any():
                vmin, vmax = dep_peak_np[mask_peak_valid].min(), dep_peak_np[mask_peak_valid].max()
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                dep_color_peak = np.ones((h, w, 3), dtype=np.uint8) * 255
                dep_color_peak[mask_peak_valid] = (cmap(norm(dep_peak_np[mask_peak_valid]))[:, :3] * 255).astype(np.uint8)
            else:
                dep_color_peak = np.ones((h, w, 3), dtype=np.uint8) * 255
            imageio.imwrite(os.path.join(out_dir, f'{idx:04d}_depth_peak.png'), dep_color_peak)

        return {
            'depth_expected': depth_expected.detach().cpu(),
            'depth_peak': depth_peak.detach().cpu(),
            'valid_peak': valid_peak.detach().cpu(),
            'camera_mat': camera_mat.detach().cpu(),
            'scale_mat': scale_mat.detach().cpu(),
            'c2w': c2w.detach().cpu() if torch.is_tensor(c2w) else c2w,
            'img_idx': img_idx.item() if torch.is_tensor(img_idx) else img_idx,
            'h': h, 'w': w
        }

    @torch.no_grad()
    @torch.no_grad()
    def eval_global_matching_every_50k(self, test_loader, it, out_dir,
                                       neighbor_threshold=2, angle_threshold=90.0):
        """
        Global matching evaluation following the paper (supports both peak and expected depths):
        1. Extract Peak/Expected 3D point clouds for each frame, sampled only at peak-valid
           pixels (ensuring that the points lie around the object).
        2. For frame pairs with |i-j| >= neighbor_threshold and viewing-angle difference <= 90 deg:
           - nearest-neighbor search in image space;
           - 3D Euclidean distance threshold = 0.02 * scene diagonal.
        3. Generate peak_to_peak and expected_to_expected statistics and visualizations separately.
        """
        from model.common import arange_pixels, transform_to_world, project_to_cam
        from torch.nn import functional as F

        os.makedirs(out_dir, exist_ok=True)
        self.model.eval()
        if self.pose_param_net:
            self.pose_param_net.eval()

        depth_dir = os.path.join(out_dir, 'depth_maps')
        os.makedirs(depth_dir, exist_ok=True)
        all_frames = []
        test_images = []
        test_data_list = list(test_loader)

        for data in test_data_list:
            bs, _, h, w = data['img'].shape
            info = self.extract_dual_depths(data, (h, w), it, out_dir=depth_dir)
            all_frames.append(info)
            img_np = data['img'].squeeze(0).permute(1, 2, 0).cpu().numpy()
            test_images.append(img_np)

        n = len(all_frames)
        if n < 2:
            print("[GlobalMatch] Not enough test frames; skipping evaluation.")
            return {}

        c2ws = torch.stack([f['c2w'] for f in all_frames]).to(self.device)
        centers = c2ws[:, :3, 3]
        scene_center = centers.mean(dim=0)
        pos_vecs = centers - scene_center.unsqueeze(0)
        pos_norm = F.normalize(pos_vecs, dim=-1)

        pixel_cache = {}
        for idx in range(n):
            h, w = all_frames[idx]['h'], all_frames[idx]['w']
            _, pix = arange_pixels(resolution=(h, w), device=self.device)
            pixel_cache[idx] = pix.to(self.device)

        # Build both peak and expected 3D point clouds, filtered by peak-valid pixels
        P_world_peak_list = []
        P_world_exp_list = []
        pix_valid_list = []
        for idx in range(n):
            valid = all_frames[idx]['valid_peak'].to(self.device).view(-1)  # (H*W,)
            depth_peak_flat = all_frames[idx]['depth_peak'].to(self.device).view(-1)
            depth_exp_flat = all_frames[idx]['depth_expected'].to(self.device).view(-1)

            if valid.sum() == 0:
                P_world_peak_list.append(torch.zeros((0, 3), device=self.device))
                P_world_exp_list.append(torch.zeros((0, 3), device=self.device))
                pix_valid_list.append(torch.zeros((0, 2), device=self.device))
                continue

            pix = pixel_cache[idx][0, valid, :]  # (N_valid, 2)

            cam = all_frames[idx]['camera_mat'].to(self.device)
            c2w = c2ws[idx].unsqueeze(0)

            # Peak point cloud
            dep_peak = depth_peak_flat[valid].view(1, -1, 1)
            P_cam_peak = transform_to_world(pix.unsqueeze(0), dep_peak, cam)
            P_world_peak = P_cam_peak @ c2w[:, :3, :3].transpose(1, 2) + c2w[:, :3, 3]
            P_world_peak_list.append(P_world_peak.squeeze(0))

            # Expected point cloud (also sampled only near the object surface)
            dep_exp = depth_exp_flat[valid].view(1, -1, 1)
            P_cam_exp = transform_to_world(pix.unsqueeze(0), dep_exp, cam)
            P_world_exp = P_cam_exp @ c2w[:, :3, :3].transpose(1, 2) + c2w[:, :3, 3]
            P_world_exp_list.append(P_world_exp.squeeze(0))

            pix_valid_list.append(pix)

        # Compute the scene diagonal to obtain the 3D distance threshold
        non_empty = [p for p in P_world_peak_list if p.numel() > 0]
        if len(non_empty) == 0:
            print("[GlobalMatch] No valid peak points in any frame.")
            return {}
        all_pts = torch.cat(non_empty, dim=0)
        bbox_min = all_pts.min(dim=0)[0]
        bbox_max = all_pts.max(dim=0)[0]
        scene_diagonal = torch.norm(bbox_max - bbox_min).item()
        threshold_3d = 0.02 * scene_diagonal
        print(f"[GlobalMatch] Scene diagonal: {scene_diagonal:.4f}, 3D threshold (0.02x): {threshold_3d:.4f}")

        stats = {
            'peak_to_peak': {'pairs': 0, 'total': 0, 'matched': 0},
            'expected_to_expected': {'pairs': 0, 'total': 0, 'matched': 0},
        }

        vis_dir = os.path.join(out_dir, 'match_vis')
        os.makedirs(vis_dir, exist_ok=True)
        vis_count = 0
        max_vis_pairs = 20  # shared upper limit for both modes

        def batch_nearest_neighbor(query_pts, ref_pts, batch_size=2048):
            M = query_pts.shape[0]
            indices = torch.zeros(M, dtype=torch.long, device=query_pts.device)
            for i in range(0, M, batch_size):
                end = min(i + batch_size, M)
                dist = torch.cdist(query_pts[i:end], ref_pts, p=2)
                indices[i:end] = dist.argmin(dim=1)
            return indices

        for i in range(n):
            h_i, w_i = all_frames[i]['h'], all_frames[i]['w']
            pix_i = pix_valid_list[i]
            if pix_i.shape[0] == 0:
                continue

            # Match peak and expected separately
            for mode_i, P_i in [('peak', P_world_peak_list[i]), ('expected', P_world_exp_list[i])]:
                if P_i.shape[0] == 0:
                    continue

                for j in range(n):
                    if i == j or abs(i - j) < neighbor_threshold:
                        continue

                    cos_ang = torch.dot(pos_norm[i], pos_norm[j]).item()
                    if cos_ang < 0.0:
                        continue

                    for mode_j, P_j in [('peak', P_world_peak_list[j]), ('expected', P_world_exp_list[j])]:
                        if mode_i != mode_j:  # only compare the same depth type
                            continue
                        if P_j.shape[0] == 0:
                            continue

                        key = f'{mode_i}_to_{mode_j}'
                        h_j, w_j = all_frames[j]['h'], all_frames[j]['w']
                        pix_j = pix_valid_list[j]
                        cam_j = all_frames[j]['camera_mat'].to(self.device)
                        c2w_j = c2ws[j].unsqueeze(0)

                        # Reproject P_i into frame j
                        w2c_j = torch.inverse(c2w_j)
                        P_cam_j = P_i @ w2c_j[:, :3, :3].transpose(1, 2) + w2c_j[:, :3, 3]
                        p_j_2d, valid_proj = project_to_cam(P_cam_j, cam_j, self.device)
                        valid_proj = valid_proj.bool().squeeze()

                        if valid_proj.sum() == 0:
                            continue

                        P_i_valid = P_i[valid_proj]
                        pix_i_valid = pix_i[valid_proj]
                        p_j_2d_valid = p_j_2d[0, valid_proj, :]

                        # Nearest-neighbor search in the image space of frame j
                        nearest_idx = batch_nearest_neighbor(p_j_2d_valid, pix_j)
                        P_j_matched = P_j[nearest_idx]

                        # 3D Euclidean distance (the basis of Eq. 8 in the paper)
                        dist_3d = torch.norm(P_i_valid - P_j_matched, dim=-1)
                        matched_mask = dist_3d < threshold_3d

                        total = matched_mask.numel()
                        matched = matched_mask.sum().item()

                        stats[key]['pairs'] += 1
                        stats[key]['total'] += total
                        stats[key]['matched'] += matched

                        # Visualization (sparsely sampled to avoid clutter)
                        if vis_count < max_vis_pairs and total > 0:
                            u_src = ((pix_i_valid[:, 0] + 1) / 2 * (w_i - 1)).cpu().numpy()
                            v_src = ((pix_i_valid[:, 1] + 1) / 2 * (h_i - 1)).cpu().numpy()
                            u_tgt = ((pix_j[nearest_idx, 0] + 1) / 2 * (w_j - 1)).cpu().numpy()
                            v_tgt = ((pix_j[nearest_idx, 1] + 1) / 2 * (h_j - 1)).cpu().numpy()

                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
                            ax1.imshow(test_images[i])
                            ax2.imshow(test_images[j])
                            ax1.set_title(f'Src {i} ({mode_i})')
                            ax2.set_title(f'Tgt {j} ({mode_j})')
                            ax1.axis('off')
                            ax2.axis('off')

                            step = max(1, total // 150)
                            for k in range(0, total, step):
                                is_ok = matched_mask[k].item()
                                color = 'lime' if is_ok else 'red'
                                ax1.plot(u_src[k], v_src[k], 'o', color=color, markersize=2, alpha=0.7)
                                ax2.plot(u_tgt[k], v_tgt[k], 'o', color=color, markersize=2, alpha=0.7)
                                con = ConnectionPatch(
                                    xyA=(u_tgt[k], v_tgt[k]), xyB=(u_src[k], v_src[k]),
                                    coordsA="data", coordsB="data",
                                    axesA=ax2, axesB=ax1, color=color, alpha=0.3, linewidth=0.5
                                )
                                ax2.add_artist(con)

                            plt.tight_layout()
                            vis_path = os.path.join(vis_dir, f'{mode_i}_pair{i}_to_{j}_it{it}.png')
                            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                            plt.close()
                            vis_count += 1

        summary = {}
        log_lines = []
        log_lines.append(f"=== Global Depth Matching Evaluation @ iteration {it} ===")
        log_lines.append(f"Test frames: {n}, Neighbor threshold: {neighbor_threshold}, Angle threshold: {angle_threshold}°")
        log_lines.append(f"Scene diagonal: {scene_diagonal:.4f}, 3D threshold (0.02x): {threshold_3d:.4f}")

        for key, s in stats.items():
            pairs = s['pairs']
            total = s['total']
            matched = s['matched']
            ratio = matched / total if total > 0 else 0.0
            summary[key] = {
                'total_pairs': pairs,
                'total_valid': total,
                'total_matched': matched,
                'avg_ratio': ratio
            }
            log_lines.append(f"[{key}] valid_pairs={pairs}, total_pts={total}, matched={matched}, ratio={ratio:.4f}")

        log_path = os.path.join(out_dir, f'global_matching_it{it}.txt')
        with open(log_path, 'w') as f:
            f.write('\n'.join(log_lines) + '\n')

        print(f"[GlobalMatch] Results saved to {log_path}")
        for line in log_lines:
            print(line)

        return summary

    def compute_loss(self, data, eval_mode=False, it=None, epoch=None, scheduling_start=None, out_render_path=None,
                     fxfy=None, T=None):
        weights = {}
        weights_name_list = ['pc_weight', 'rgb_s_weight', 'depth_consistency_weight',
                             'weight_dist_2nd_loss', 'weight_dist_1st_loss']
        weights_list = [
            self.anneal(getattr(self, w)[0], getattr(self, w)[1], scheduling_start, self.annealing_epochs, epoch) for w
            in weights_name_list]
        for i, weight in enumerate(weights_list):
            weights[weights_name_list[i]] = weight

        current_alpha = self.get_sigmoid_alpha_by_iter(it, end_it=100000, steepness=6.0)
        weights['rgb_weight'] = self.rgb_weight_cfg[0]
        base_depth_scale = self.depth_weight_cfg[0]
        weights['depth_weight'] = (1.0 - current_alpha) * base_depth_scale

        rgb_loss_type = 'l1' if epoch < self.annealing_epochs + scheduling_start else 'l2'
        render_model = (weights['rgb_weight'] != 0.0) or (weights['depth_weight'] != 0.0)
        use_ref_imgs = ((weights['pc_weight'] != 0.0) or (weights['rgb_s_weight'] != 0.0))

        n_points = self.n_training_points
        nl = self.nearest_limit
        (img, depth_input, camera_mat_gt, scale_mat, img_idx) = self.process_data_dict(data)
        if use_ref_imgs:
            (ref_img, depth_ref, ref_idx) = self.process_data_reference(data)

        device = self.device
        batch_size, _, h, w = img.shape
        batch_size, _, h_depth, w_depth = depth_input.shape
        kwargs = dict()
        kwargs['t_list'] = self.pose_param_net.get_t()
        kwargs['weights'] = weights
        kwargs['rgb_loss_type'] = rgb_loss_type

        if self.pose_param_net is not None:
            num_cams = self.pose_param_net.num_cams
            c2w = self.pose_param_net(img_idx)
            world_mat = torch.inverse(c2w).unsqueeze(0)

        if self.distortion_net is not None:
            scale_input, shift_input = self.distortion_net(img_idx)
            epsilon = 1e-6
            disp_input = 1.0 / (depth_input + epsilon)
            if self.shift_first:
                disp_input = (disp_input + shift_input) * scale_input
            else:
                disp_input = disp_input * scale_input + shift_input
            disp_input = torch.clamp(disp_input, min=epsilon)
            depth_input = 1.0 / disp_input

        if self.optimizer_focal:
            fxfy = self.focal_net(0)
            pad = torch.zeros(4).to(device)
            one = torch.tensor([1]).to(device)
            camera_mat = torch.cat([fxfy[0:1], pad, -fxfy[1:2], pad, -one, pad, one])
            camera_mat = camera_mat.view(1, 4, 4)
        else:
            camera_mat = camera_mat_gt

        ray_idx = torch.randperm(h * w, device=device)[:n_points]
        img_flat = img.view(batch_size, 3, h * w).permute(0, 2, 1)
        rgb_gt = img_flat[:, ray_idx]
        p_full = arange_pixels((h, w), batch_size, device=device)[1]
        p = p_full[:, ray_idx]
        pix = ray_idx

        if render_model:
            out_dict = self.model(
                p, pix, camera_mat, world_mat, scale_mat,
                self.rendering_technique, it=it,
                eval_mode=eval_mode, depth_img=depth_input,
                img_size=(h, w))
            rendered_rgb = out_dict['rgb']
            rendered_depth = out_dict['depth_pred']
            gt_depth = out_dict['depth_gt']
        else:
            rendered_rgb = None
            rendered_depth = None
            gt_depth = None

        loss_global = None
        if (not eval_mode and it is not None and render_model
                and self.train_dataset is not None
                and self.pose_param_net is not None
                and it >= self.eggm_start_it):
            # refresh the cached peak anchors at the scheduled milestones
            # (paper: matching points are updated every 50k iterations)
            if (it in self.eggm_update_iters and self._eggm_cache_it != it) \
                    or self._eggm_cache is None:
                self.update_eggm_anchor_cache(it)
            if self._eggm_cache is not None:
                # linear warm-up of the global weight after activation
                w_g = min(1.0, (it - self.eggm_start_it)
                          / float(max(1, self.eggm_warmup_iters))) \
                    * self.eggm_warmup_weight
                lg = self.compute_global_loss(img_idx, out_dict, p,
                                              camera_mat, device)
                loss_global = w_g * lg

        if use_ref_imgs:
            c2w_ref = self.pose_param_net(ref_idx)
            if self.distortion_net is not None:
                scale_ref, shift_ref = self.distortion_net(ref_idx)
                epsilon = 1e-6
                disp_ref = 1.0 / (depth_ref + epsilon)
                if self.shift_first:
                    disp_ref = scale_ref * (disp_ref + shift_ref)
                else:
                    disp_ref = scale_ref * disp_ref + shift_ref
                disp_ref = torch.clamp(disp_ref, min=epsilon)
                depth_ref = 1.0 / disp_ref

            if self.detach_ref_img:
                c2w_ref = c2w_ref.detach()
                scale_ref = scale_ref.detach()
                shift_ref = shift_ref.detach()
                depth_ref = depth_ref.detach()
            ref_Rt = torch.inverse(c2w_ref).unsqueeze(0)

            if img_idx < (num_cams - 1):
                d1 = depth_input
                d2 = depth_ref
                img1 = img
                img2 = ref_img
                Rt_rel_12 = ref_Rt @ torch.inverse(world_mat)
                R_rel_12 = Rt_rel_12[:, :3, :3]
                t_rel_12 = Rt_rel_12[:, :3, 3]
                scale2 = scale_ref
            else:
                d1 = depth_ref
                d2 = depth_input
                img1 = ref_img
                img2 = img
                Rt_rel_12 = world_mat @ torch.inverse(ref_Rt)
                R_rel_12 = Rt_rel_12[:, :3, :3]
                t_rel_12 = Rt_rel_12[:, :3, 3]
                scale2 = scale_input

            ratio = self.pc_ratio
            sample_resolution = (int(h_depth / ratio), int(w_depth / ratio))
            pixel_locations, p_pc = arange_pixels(resolution=sample_resolution, device=device)
            d1 = F.interpolate(d1, sample_resolution, mode='nearest')
            d2 = F.interpolate(d2, sample_resolution, mode='nearest')
            d1[d1 < nl] = nl
            d2[d2 < nl] = nl
            pc1 = transform_to_world(p_pc, d1.view(1, -1, 1), camera_mat)
            pc2 = transform_to_world(p_pc, d2.view(1, -1, 1), camera_mat)

            if weights['rgb_s_weight'] != 0.0:
                img1 = F.interpolate(img1, sample_resolution, mode='bilinear')
                img2 = F.interpolate(img2, sample_resolution, mode='bilinear')
                rgb_pc1 = get_tensor_values(img1, p_pc, mode='bilinear', scale=False, detach=False, detach_p=False,
                                            align_corners=True)
                if self.detach_rgbs_scale:
                    pc1_ = pc1.detach().clone()
                    pc1_rotated = pc1_ @ R_rel_12.transpose(1, 2) + t_rel_12
                else:
                    pc1_rotated = pc1 @ R_rel_12.transpose(1, 2) + t_rel_12
                mask_pc1_invalid = (-pc1_rotated[:, :, 2:] < nl).expand_as(pc1_rotated)
                pc1_rotated[mask_pc1_invalid] = nl
                p_reprojected, valid_mask = project_to_cam(pc1_rotated, camera_mat, device)
                rgb_pc1_proj = get_tensor_values(img2, p_reprojected, mode='bilinear', scale=False, detach=False,
                                                 detach_p=False, align_corners=True)
                rgb_pc1 = rgb_pc1.view(batch_size, sample_resolution[0], sample_resolution[1], 3)
                rgb_pc1_proj = rgb_pc1_proj.view(batch_size, sample_resolution[0], sample_resolution[1], 3)
                valid_mask = valid_mask.view(batch_size, sample_resolution[0], sample_resolution[1], 1)
                kwargs['rgb_pc1'] = rgb_pc1
                kwargs['rgb_pc1_proj'] = rgb_pc1_proj
                kwargs['valid_points'] = valid_mask
                if (it % self.vis_reprojection_every) == 0:
                    Image.fromarray(
                        ((rgb_pc1[0] * 255).detach().cpu().numpy()).astype(np.uint8)
                    ).convert("RGB").save(
                        os.path.join(out_render_path, '%d_%04d_img1.png' % (it, img_idx))
                    )
                    Image.fromarray(
                        ((rgb_pc1_proj[0] * 255).detach().cpu().numpy()).astype(np.uint8)
                    ).convert("RGB").save(
                        os.path.join(out_render_path, '%d_%04d_img2.png' % (it, img_idx))
                    )
            pc1 = pc1 @ R_rel_12.transpose(1, 2) + t_rel_12
            if self.scale_pcs:
                pc1 = pc1 / scale2
                pc2 = pc2 / scale2

            kwargs['X'] = pc1
            kwargs['Y'] = pc2

            kwargs['sample_resolution'] = sample_resolution
            kwargs['p_2d'] = pixel_locations

        if render_model and self.detach_gt_depth:
            gt_depth = gt_depth.detach()
        loss_dict = self.loss(rendered_rgb, rgb_gt, rendered_depth, gt_depth, loss_global=loss_global, **kwargs)
        if self.optimizer_focal:
            loss_dict['focalx'] = fxfy[0] / camera_mat_gt[0, 0, 0]
            loss_dict['focaly'] = fxfy[1] / camera_mat_gt[0, 1, 1]

        loss_dict['scale'] = scale_input
        loss_dict['shift'] = shift_input
        return loss_dict
