import os
import re
import sys
import torch
import torch.nn as nn
import logging
import traceback
from model.losses import Loss
import numpy as np
from PIL import Image
import imageio
from torch.nn import functional as F
from model.common import (
    get_tensor_values,
    arange_pixels, project_to_cam, transform_to_world,
)

logger_py = logging.getLogger(__name__)

## NoPe-NeRF++ (Sec 4.1): SuperPoint + SuperGlue lazy import flag
SP_SG_AVAILABLE = False
Matching = None
frame2tensor = None


def _try_import_superglue():
    """Lazily import SuperGlue to avoid circular imports; print detailed traceback on failure"""
    global SP_SG_AVAILABLE, Matching, frame2tensor
    if SP_SG_AVAILABLE:
        return True
    try:
        import importlib
        _matching_mod = importlib.import_module('models.matching')
        _utils_mod = importlib.import_module('models.utils')
        Matching = _matching_mod.Matching
        frame2tensor = _utils_mod.frame2tensor
        SP_SG_AVAILABLE = True
        logger_py.info('NoPe-NeRF++: SuperPoint + SuperGlue imported successfully.')
        return True
    except Exception as e:
        tb_str = traceback.format_exc()
        logger_py.warning(
            'NoPe-NeRF++: Cannot import models.matching / models.utils. '
            'Will fallback to OpenCV SIFT.\n'
            'Most likely cause: circular import due to incorrect '
            '"from model import ..." or "import model" inside models/*.py '
            '(check models/matching.py, models/utils.py, etc.)\n'
            'Detailed error:\n%s' % tb_str
        )
        return False


## End NoPe-NeRF++ (Sec 4.1)

## NoPe-NeRF++ (Sec 5.1): OpenCV fallback detection
CV2_AVAILABLE = False
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    logger_py.warning('NoPe-NeRF++: OpenCV (cv2) not found. Track extraction will be disabled.')


## End NoPe-NeRF++ (Sec 5.1)


## NoPe-NeRF++ (Sec 4.1): Union-Find for building global feature tracks
class UnionFindDict:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


## End NoPe-NeRF++ (Sec 4.1)


class Trainer(object):
    def __init__(self, model, optimizer, cfg, device=None, optimizer_pose=None, pose_param_net=None,
                 optimizer_focal=None, focal_net=None, optimizer_distortion=None, distortion_net=None,
                 cfg_all=None, **kwargs):
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

        ## NoPe-NeRF++: keep a global cfg reference for reading dataloading / image_dir etc.
        self.cfg_all = cfg_all if cfg_all is not None else cfg
        ## End NoPe-NeRF++

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

        self.pc_weight = cfg['pc_weight']
        self.rgb_s_weight = cfg['rgb_s_weight']
        self.rgb_weight = cfg['rgb_weight']
        self.depth_weight = cfg['depth_weight']
        self.weight_dist_2nd_loss = cfg['weight_dist_2nd_loss']
        self.weight_dist_1st_loss = cfg['weight_dist_1st_loss']
        self.depth_consistency_weight = cfg['depth_consistency_weight']

        ## NoPe-NeRF++ (Sec 5.1): staged parameters given in the paper (defaults strictly follow the paper)
        self.local_phase_steps = cfg.get('local_phase_steps', 40000)
        self.global_phase_steps_first = cfg.get('global_phase_steps_first', 10000)
        self.global_phase_steps_second = cfg.get('global_phase_steps_second', 5000)

        self.lambda_depth_local = cfg.get('lambda_depth_local', 0.04)
        self.lambda_mgc_local = cfg.get('lambda_mgc_local', 1.0)  # Paper default: 1.0
        self.lambda_track_global_first = cfg.get('lambda_track_global_first', 10.0)
        self.lambda_track_global_second = cfg.get('lambda_track_global_second', 1.0)
        ## End NoPe-NeRF++ (Sec 5.1)

        ## NoPe-NeRF++ (Sec 4.1/4.3): Association & Global Tracks
        self.n_tracks = cfg.get('n_tracks', 10000)
        self.tracks_dir = cfg.get('tracks_dir', './tracks')
        os.makedirs(self.tracks_dir, exist_ok=True)

        # Local phase buffers
        self.local_poses_buffer = {}
        self.local_depths_buffer = {}
        self.local_cams_buffer = {}
        self.tracks_cache = {}
        self.tracks_loaded = False
        self.tracks_extracted_this_run = False
        self.delayed_collection = False

        # Global track points Q (nn.Parameter, optimized in global phase)
        self.track_pts_3d = None
        self.tracks_observations = []
        self.track_lr = cfg.get('track_lr', cfg.get('pose_lr', 0.0005))
        self.n_track_samples = cfg.get('n_track_samples', 512)  # FIX-C: reduced from 2048 to 512 for 5x speedup

        # Frame cache for random matching scheme (local phase)
        self.frame_img_cache = {}
        self.frame_dpt_cache = {}
        self.frame_cache_full = False

        # SuperPoint+SuperGlue / SIFT lazy init
        self._matching = None
        self._cv_sift = None

        # FIX-D: if the passed pose_param_net has requires_grad=False, force-enable it
        # (parameters may be frozen in some checkpoint loading scenarios)
        if self.pose_param_net is not None:
            for p in self.pose_param_net.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    print(f'[NoPe-NeRF++ FIX-D] Constructor: pose param requires_grad was False, set to True.')

        ## NoPe-NeRF++ (Sec 4.1): image_dir inference and natural sorting of image_list
        self.image_dir = cfg.get('image_dir', None)
        if self.image_dir is None or not os.path.exists(self.image_dir):
            dl_cfg = self.cfg_all.get('dataloading', {}) if self.cfg_all else {}
            base = (dl_cfg.get('path', '') or '')
            scene = (dl_cfg.get('scene', '') or '')
            if isinstance(scene, (list, tuple)) and len(scene) > 0:
                scene = str(scene[0])
            elif not isinstance(scene, str):
                scene = str(scene)

            candidates = []
            if base and scene:
                candidates.extend([
                    os.path.join(base, scene, 'images'),
                    os.path.join(base, scene),
                ])
            if base:
                candidates.extend([
                    os.path.join(base, 'images'),
                    base,
                ])
            if scene:
                candidates.extend([
                    './data/nerf_llff_data/{}/images'.format(scene),
                    './data/{}/images'.format(scene),
                    '../data/{}/images'.format(scene),
                ])

            for c in candidates:
                if c and os.path.exists(c) and os.path.isdir(c):
                    imgs = [f for f in os.listdir(c)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if len(imgs) > 0:
                        self.image_dir = c
                        break

        if self.image_dir is None or not os.path.exists(self.image_dir):
            logger_py.warning(
                'NoPe-NeRF++: image_dir not found. MST/track extraction will fail.\n'
                'Please set training.image_dir in your YAML.'
            )
        else:
            logger_py.info(f'NoPe-NeRF++: Auto-detected image_dir = {self.image_dir}')

        if self.image_dir is not None and os.path.exists(self.image_dir):
            raw_list = [f for f in os.listdir(self.image_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))]

            def _nat_key(s):
                return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

            self.image_list = sorted(raw_list, key=_nat_key)
            print(
                f'[NoPe-NeRF++ INIT] image_dir={self.image_dir}, image_list (first 5)={self.image_list[:5]}, total={len(self.image_list)}')
            # Get image size (used for reprojection)
            path0 = os.path.join(self.image_dir, self.image_list[0])
            img0 = cv2.imread(path0) if CV2_AVAILABLE else None
            if img0 is not None:
                self.img_h, self.img_w = img0.shape[:2]
            else:
                self.img_h, self.img_w = 800, 800
        else:
            self.image_list = []
            self.img_h, self.img_w = 800, 800
            print(f'[NoPe-NeRF++ INIT] ERROR: image_dir invalid or not found: {self.image_dir}')
        ## End NoPe-NeRF++ (Sec 4.1)

        self.loss = Loss(cfg)

        ## NoPe-NeRF++ (Sec 4.1): Build Maximum Spanning Tree (MST) at init
        self.mst_edges = []
        self.mst_adj = {}
        self.mst_parent = {}
        self.mst_children = {}
        self.pair_matches = {}  # (i,j) -> (mkpts_i, mkpts_j, idx_i, idx_j)
        self.random_match_candidates = {}
        self.all_kps = []
        self.all_des = []
        self.association_ready = False
        self.random_match_min_pairs = cfg.get('random_match_min_pairs', 100)

        if len(self.image_list) > 0:
            self._build_association_mst(self.device)
        else:
            logger_py.warning('NoPe-NeRF++: No images found. MST skipped.')
        ## End NoPe-NeRF++ (Sec 4.1)

        ## NoPe-NeRF++ (Sec 4.3): Try load pre-computed tracks (optional)
        self._load_tracks_from_disk()
        if self.tracks_loaded:
            logger_py.info(f'=> NoPe-NeRF++: Resume with {len(self.tracks_cache) // 2} track pairs pre-loaded.')
        else:
            logger_py.info(f'=> NoPe-NeRF++: No pre-computed tracks. Will extract at step {self.local_phase_steps}.')
        ## End NoPe-NeRF++ (Sec 4.3)

        ## NoPe-NeRF++ (Sec 4.2): ICP initialization flag
        self.icp_init_done = False
        ## End NoPe-NeRF++

    ## NoPe-NeRF++ (Sec 4.1): MST dispatcher - prefer SuperPoint+SuperGlue, fall back to SIFT
    def _build_association_mst(self, device):
        if _try_import_superglue():
            try:
                self._build_association_mst_superglue(device)
                return
            except Exception as e:
                logger_py.error(f'NoPe-NeRF++: SuperPoint+SuperGlue MST failed: {e}. Falling back to SIFT.')
        if CV2_AVAILABLE:
            logger_py.warning('NoPe-NeRF++: Falling back to OpenCV SIFT for MST.')
            self._build_association_mst_sift(device)
        else:
            logger_py.error('NoPe-NeRF++: No feature extraction backend available. MST not built.')

    ## End NoPe-NeRF++ (Sec 4.1)

    ## NoPe-NeRF++ (Sec 4.1): Build MST using SuperPoint + SuperGlue
    def _build_association_mst_superglue(self, device):
        print('[NoPe-NeRF++ MST] Building Maximum Spanning Tree from SuperPoint+SuperGlue...')
        self._init_superglue(device)
        num_imgs = len(self.image_list)

        # 1. Extract SuperPoint features for all images
        features = []
        for img_name in self.image_list:
            path = os.path.join(self.image_dir, img_name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                features.append(None)
                continue
            inp = frame2tensor(img, device)  # (1, 1, H, W)
            with torch.no_grad():
                pred = self._matching.superpoint({'image': inp})
            # pred: {'keypoints': [(N,2)], 'scores': [(N,)], 'descriptors': [(256,N)]}
            features.append({
                'image': inp,
                'keypoints': pred['keypoints'][0],  # (N, 2)
                'scores': pred['scores'][0],  # (N,)
                'descriptors': pred['descriptors'][0],  # (256, N)
            })

        # 2. Compute all-pair SuperGlue matches
        edges = []  # (weight, i, j, mkpts_i, mkpts_j, idx_i, idx_j)
        for i in range(num_imgs):
            for j in range(i + 1, num_imgs):
                if features[i] is None or features[j] is None:
                    continue

                data = {
                    'image0': features[i]['image'],
                    'image1': features[j]['image'],
                    'keypoints0': features[i]['keypoints'][None],  # (1, N, 2)
                    'scores0': features[i]['scores'][None],  # (1, N)
                    'descriptors0': features[i]['descriptors'][None],  # (1, 256, N)
                    'keypoints1': features[j]['keypoints'][None],
                    'scores1': features[j]['scores'][None],
                    'descriptors1': features[j]['descriptors'][None],
                }

                with torch.no_grad():
                    pred_sg = self._matching.superglue(data)

                matches = pred_sg['matches0'][0].cpu().numpy()  # (N_i,)
                valid = matches > -1
                if valid.sum() == 0:
                    continue

                mkpts_i = features[i]['keypoints'][valid].cpu().numpy()  # (M, 2)
                mkpts_j = features[j]['keypoints'][matches[valid]].cpu().numpy()  # (M, 2)
                idx_i = np.where(valid)[0].astype(np.int32)
                idx_j = matches[valid].astype(np.int32)

                edges.append((int(valid.sum()), i, j, mkpts_i, mkpts_j, idx_i, idx_j))

        if len(edges) == 0:
            logger_py.warning('NoPe-NeRF++: No SuperGlue matches found. MST not built.')
            return

        # 3. Kruskal MST
        edges.sort(reverse=True, key=lambda x: x[0])
        uf_mst = UnionFindDict()
        for idx in range(num_imgs):
            uf_mst.find(idx)
        mst_edges = []
        self.mst_adj = {idx: [] for idx in range(num_imgs)}
        for w, i, j, mk_i, mk_j, idx_i, idx_j in edges:
            if uf_mst.find(i) != uf_mst.find(j):
                uf_mst.union(i, j)
                mst_edges.append((i, j, w, mk_i, mk_j, idx_i, idx_j))
                self.mst_adj[i].append((j, w))
                self.mst_adj[j].append((i, w))
                if len(mst_edges) == num_imgs - 1:
                    break

        self.mst_edges = mst_edges
        print(f'[NoPe-NeRF++ MST] Built MST with {len(mst_edges)} edges (SuperPoint+SuperGlue).')

        # 4. Determine parent/children (DFS from root 0)
        self.mst_parent = {}
        self.mst_children = {idx: [] for idx in range(num_imgs)}
        visited = [False] * num_imgs
        stack = [0]
        visited[0] = True
        while stack:
            cur = stack.pop()
            for nei, w in self.mst_adj[cur]:
                if not visited[nei]:
                    visited[nei] = True
                    self.mst_parent[nei] = cur
                    self.mst_children[cur].append(nei)
                    stack.append(nei)

        # 5. Store MST edge matches
        self.pair_matches = {}
        for i, j, w, mk_i, mk_j, idx_i, idx_j in mst_edges:
            self.pair_matches[(i, j)] = (mk_i, mk_j, idx_i, idx_j)
            self.pair_matches[(j, i)] = (mk_j, mk_i, idx_j, idx_i)

        # 6. Store random matching candidates (non-MST pairs with >=100 matches)
        self.random_match_candidates = {idx: [] for idx in range(num_imgs)}
        for w, i, j, mk_i, mk_j, idx_i, idx_j in edges:
            if w >= self.random_match_min_pairs:
                if (i, j) not in self.pair_matches and (j, i) not in self.pair_matches:
                    self.random_match_candidates[i].append((j, w, mk_i, mk_j, idx_i, idx_j))
                    self.random_match_candidates[j].append((i, w, mk_j, mk_i, idx_j, idx_i))

        # Cache keypoints for subsequent track extraction / ICP
        self.all_kps = [f['keypoints'].cpu().numpy() if f is not None else np.zeros((0, 2), dtype=np.float32) for f in
                        features]
        self.all_des = [f['descriptors'].cpu().numpy() if f is not None else None for f in features]
        self.association_ready = True
        logger_py.info(f'NoPe-NeRF++: Association init complete (SP+SG). MST edges={len(mst_edges)}.')

    ## End NoPe-NeRF++ (Sec 4.1)

    ## NoPe-NeRF++ (Sec 4.1): Build MST using OpenCV SIFT + Kruskal (Fallback)
    def _build_association_mst_sift(self, device):
        print('[NoPe-NeRF++ MST] Building Maximum Spanning Tree from SIFT matches (fallback)...')
        self._init_cv_sift()
        num_imgs = len(self.image_list)
        all_kps = []
        all_des = []

        # 1. Extract keypoints for all images
        for img_name in self.image_list:
            path = os.path.join(self.image_dir, img_name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                all_kps.append(np.zeros((0, 2), dtype=np.float32))
                all_des.append(None)
                continue
            kp, des = self._cv_sift.detectAndCompute(img, None)
            if kp is None or len(kp) == 0:
                all_kps.append(np.zeros((0, 2), dtype=np.float32))
                all_des.append(None)
                continue
            pts = np.float32([p.pt for p in kp])
            all_kps.append(pts)
            all_des.append(des)

        # 2. Compute all-pair matching counts
        edges = []  # (weight, i, j, mkpts_i, mkpts_j, idx_i, idx_j)
        for i in range(num_imgs):
            for j in range(i + 1, num_imgs):
                des_i, des_j = all_des[i], all_des[j]
                if des_i is None or des_j is None:
                    continue
                bf = cv2.BFMatcher(cv2.NORM_L2)
                matches = bf.knnMatch(des_i, des_j, k=2)
                good = []
                mkpts_i_idx = []
                mkpts_j_idx = []
                for m_n in matches:
                    if len(m_n) < 2:
                        continue
                    m, n = m_n[0], m_n[1]
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                        mkpts_i_idx.append(m.queryIdx)
                        mkpts_j_idx.append(m.trainIdx)
                if len(good) > 0:
                    edges.append((len(good), i, j,
                                  all_kps[i][mkpts_i_idx],
                                  all_kps[j][mkpts_j_idx],
                                  np.array(mkpts_i_idx),
                                  np.array(mkpts_j_idx)))

        if len(edges) == 0:
            logger_py.warning('NoPe-NeRF++: No SIFT matches found. MST not built.')
            return

        # 3. Kruskal MST
        edges.sort(reverse=True, key=lambda x: x[0])
        uf_mst = UnionFindDict()
        for idx in range(num_imgs):
            uf_mst.find(idx)
        mst_edges = []
        self.mst_adj = {idx: [] for idx in range(num_imgs)}
        for w, i, j, mk_i, mk_j, idx_i, idx_j in edges:
            if uf_mst.find(i) != uf_mst.find(j):
                uf_mst.union(i, j)
                mst_edges.append((i, j, w, mk_i, mk_j, idx_i, idx_j))
                self.mst_adj[i].append((j, w))
                self.mst_adj[j].append((i, w))
                if len(mst_edges) == num_imgs - 1:
                    break

        self.mst_edges = mst_edges
        print(f'[NoPe-NeRF++ MST] Built MST with {len(mst_edges)} edges (SIFT fallback).')

        # 4. Determine parent/children (DFS from root 0)
        self.mst_parent = {}
        self.mst_children = {idx: [] for idx in range(num_imgs)}
        visited = [False] * num_imgs
        stack = [0]
        visited[0] = True
        while stack:
            cur = stack.pop()
            for nei, w in self.mst_adj[cur]:
                if not visited[nei]:
                    visited[nei] = True
                    self.mst_parent[nei] = cur
                    self.mst_children[cur].append(nei)
                    stack.append(nei)

        # 5. Store MST edge matches
        self.pair_matches = {}
        for i, j, w, mk_i, mk_j, idx_i, idx_j in mst_edges:
            self.pair_matches[(i, j)] = (mk_i, mk_j, idx_i, idx_j)
            self.pair_matches[(j, i)] = (mk_j, mk_i, idx_j, idx_i)

        # 6. Store random matching candidates
        self.random_match_candidates = {idx: [] for idx in range(num_imgs)}
        for w, i, j, mk_i, mk_j, idx_i, idx_j in edges:
            if w >= self.random_match_min_pairs:
                if (i, j) not in self.pair_matches and (j, i) not in self.pair_matches:
                    self.random_match_candidates[i].append((j, w, mk_i, mk_j, idx_i, idx_j))
                    self.random_match_candidates[j].append((i, w, mk_j, mk_i, idx_j, idx_i))

        self.all_kps = all_kps
        self.all_des = all_des
        self.association_ready = True
        logger_py.info(f'NoPe-NeRF++: Association init complete (SIFT). MST edges={len(mst_edges)}.')

    ## End NoPe-NeRF++ (Sec 4.1)

    ## NoPe-NeRF++ (Sec 4.1): SuperPoint+SuperGlue lazy initialization
    def _init_superglue(self, device):
        if self._matching is not None:
            return
        _try_import_superglue()
        if not SP_SG_AVAILABLE:
            raise RuntimeError('SuperPoint+SuperGlue not available.')
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 4096,
            },
            'superglue': {
                'weights': 'outdoor',  # LLFF is an outdoor real-world scene, use outdoor
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        self._matching = Matching(config).eval().to(device)
        logger_py.info('NoPe-NeRF++: Matching (SuperPoint+SuperGlue) initialized.')

    ## End NoPe-NeRF++ (Sec 4.1)

    ## NoPe-NeRF++ (Sec 5.1): OpenCV SIFT fallback initialization
    def _init_cv_sift(self):
        if not CV2_AVAILABLE or self._cv_sift is not None:
            return
        self._cv_sift = cv2.SIFT_create(nfeatures=4096, contrastThreshold=0.01)

    ## End NoPe-NeRF++ (Sec 5.1)

    ## NoPe-NeRF++ (Sec 4.2): Kabsch algorithm for ICP initialization
    def _kabsch(self, P, Q):
        """
        P, Q: (3, N) numpy arrays (matched 3D points in camera space)
        Returns T (4,4) such that Q = T[:3,:3] @ P + T[:3,3:]
        """
        P_mean = P.mean(axis=1, keepdims=True)
        Q_mean = Q.mean(axis=1, keepdims=True)
        P_centered = P - P_mean
        Q_centered = Q - Q_mean
        H = P_centered @ Q_centered.T
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = Q_mean - R @ P_mean
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3:4] = t
        return T

    ## End NoPe-NeRF++ (Sec 4.2)

    ## NoPe-NeRF++ (Sec 4.2): ICP-based pose initialization via MST + DPT depth
    def _icp_init_poses(self, device):
        print('[NoPe-NeRF++ ICP] Pose initialization from MST edges (Kabsch)...')
        if not self.association_ready or len(self.mst_edges) == 0:
            logger_py.warning('NoPe-NeRF++ ICP: MST not ready. Skipping.')
            return
        if self.pose_param_net is None:
            return

        # CRITICAL-FIX: ICP initialization would overwrite the poses optimized in the checkpoint.
        # If poses were restored from checkpoint (optimizer has training history), skip ICP.
        if self.optimizer_pose is not None:
            opt_state = self.optimizer_pose.state_dict()
            if 'state' in opt_state and len(opt_state['state']) > 0:
                print('[NoPe-NeRF++ ICP] Checkpoint pose detected (optimizer has history). '
                      'Skipping ICP to preserve optimized poses.')
                self.icp_init_done = True
                return

        num_cams = self.pose_param_net.num_cams
        if len(self.local_depths_buffer) < num_cams:
            logger_py.warning(
                f'NoPe-NeRF++ ICP: Depth buffer {len(self.local_depths_buffer)}/{num_cams} not full. Skipping.')
            return

        # Intrinsics
        if len(self.local_cams_buffer) > 0:
            cam0 = list(self.local_cams_buffer.values())[0].cpu().numpy()[0]
            fx = abs(float(cam0[0, 0]))
            fy = abs(float(cam0[1, 1]))
            cx = float(cam0[0, 2]) if cam0.shape[-1] > 2 else self.img_w / 2.0
            cy = float(cam0[1, 2]) if cam0.shape[-1] > 2 else self.img_h / 2.0
        else:
            fx = fy = 800.0
            cx = self.img_w / 2.0
            cy = self.img_h / 2.0

        # BFS from root (node 0)
        root = 0
        visited = {root}
        queue = [root]

        while queue:
            cur = queue.pop(0)
            for nei, w in self.mst_adj[cur]:
                if nei in visited:
                    continue

                # FIX: MST contains all images (including test set), but the buffer only has train views.
                # Skip nodes without depth cache (test views).
                if nei not in self.local_depths_buffer or cur not in self.local_depths_buffer:
                    continue
                ## End FIX

                visited.add(nei)
                queue.append(nei)

                # Retrieve matches for this MST edge
                key = (cur, nei) if (cur, nei) in self.pair_matches else (nei, cur)
                if key not in self.pair_matches:
                    continue
                mk_cur, mk_nei, _, _ = self.pair_matches[key]

                # Get depths
                d_cur = self.local_depths_buffer[cur].detach().cpu().numpy()
                d_nei = self.local_depths_buffer[nei].detach().cpu().numpy()
                while d_cur.ndim > 2:
                    d_cur = d_cur.squeeze(0)
                while d_nei.ndim > 2:
                    d_nei = d_nei.squeeze(0)

                h_d, w_d = d_cur.shape

                # FIX: all_kps coordinates are based on the original image size; depth maps may differ in size, scale to align
                scale_x = w_d / float(self.img_w) if self.img_w > 0 else 1.0
                scale_y = h_d / float(self.img_h) if self.img_h > 0 else 1.0
                ## End FIX

                pts_cur_cam = []
                pts_nei_cam = []
                for (x_c, y_c), (x_n, y_n) in zip(mk_cur, mk_nei):
                    # FIX: apply coordinate scaling and relax the depth threshold
                    xc = int(np.clip(x_c * scale_x, 0, w_d - 1))
                    yc = int(np.clip(y_c * scale_y, 0, h_d - 1))
                    xn = int(np.clip(x_n * scale_x, 0, w_d - 1))
                    yn = int(np.clip(y_n * scale_y, 0, h_d - 1))
                    zc = d_cur[yc, xc]
                    zn = d_nei[yn, xn]
                    if zc > 1e-3 and zn > 1e-3:
                        Xc = (xc - cx) * zc / fx
                        Yc = (yc - cy) * zc / fy
                        Xn = (xn - cx) * zn / fx
                        Yn = (yn - cy) * zn / fy
                        pts_cur_cam.append([Xc, Yc, zc])
                        pts_nei_cam.append([Xn, Yn, zn])

                if len(pts_cur_cam) < 3:
                    continue

                P = np.array(pts_cur_cam, dtype=np.float32).T  # (3, N)
                Q = np.array(pts_nei_cam, dtype=np.float32).T  # (3, N)

                # T_cur_nei: transforms points from cur_cam to nei_cam
                T_cur_nei = self._kabsch(P, Q)

                # Get current world pose of cur (c2w)
                pose_cur = self.local_poses_buffer[cur].detach().cpu().numpy()  # (4,4)

                # c2w_cur = c2w_nei @ T_cur_nei  =>  c2w_nei = c2w_cur @ T_cur_nei^{-1}
                T_cur_nei_inv = np.linalg.inv(T_cur_nei)
                pose_nei = pose_cur @ T_cur_nei_inv

                # Decompose to axis-angle + translation
                R_nei = pose_nei[:3, :3]
                t_nei = pose_nei[:3, 3]
                try:
                    from scipy.spatial.transform import Rotation as R_scipy
                    r_vec = R_scipy.from_matrix(R_nei).as_rotvec()
                except Exception:
                    r_vec = np.zeros(3, dtype=np.float32)

                # Update pose_param_net
                with torch.no_grad():
                    self.pose_param_net.r[nei] = torch.from_numpy(r_vec).float().to(device)
                    self.pose_param_net.t[nei] = torch.from_numpy(t_nei).float().to(device)

        self.icp_init_done = True
        print(f'[NoPe-NeRF++ ICP] Pose initialization complete for {len(visited)} nodes.')

        # FIX-D: ensure pose parameters have requires_grad=True, otherwise ATE will never change
        if self.pose_param_net is not None:
            for p in self.pose_param_net.parameters():
                p.requires_grad_(True)

        # CRITICAL FIX: re-bind optimizer_pose parameters after ICP initialization
        # set_pose() may create new Parameters, invalidating the optimizer references
        if self.pose_param_net is not None and self.optimizer_pose is not None:
            current_params = list(self.pose_param_net.parameters())
            self.optimizer_pose.param_groups[0]['params'] = current_params

        logger_py.info(f'NoPe-NeRF++ ICP: Initialized poses via MST Kabsch.')

    ## End NoPe-NeRF++ (Sec 4.2)

    ## NoPe-NeRF++ (Sec 4.3): Extract global tracks via Union-Find + triangulation
    def _extract_global_tracks(self, device):
        print(f'[NoPe-NeRF++ TRACKS] Extracting global tracks: buffer_poses={len(self.local_poses_buffer)}')
        if not self.association_ready or len(self.mst_edges) == 0:
            logger_py.error('NoPe-NeRF++: MST not ready. Cannot extract tracks.')
            self.tracks_extracted_this_run = True
            return

        # 1. Union-Find on MST edge matches to build feature tracks
        uf = UnionFindDict()
        for (i, j), (mk_i, mk_j, idx_i, idx_j) in self.pair_matches.items():
            if i > j:
                continue
            for a, b in zip(idx_i, idx_j):
                uf.union((i, int(a)), (j, int(b)))

        tracks_dict = {}
        for key in list(uf.parent.keys()):
            root = uf.find(key)
            if root not in tracks_dict:
                tracks_dict[root] = []
            img_idx, kp_idx = key
            x, y = self.all_kps[img_idx][kp_idx]
            tracks_dict[root].append((img_idx, kp_idx, x, y))

        valid_tracks = [obs for obs in tracks_dict.values() if len(obs) >= 2]
        print(f'[NoPe-NeRF++ TRACKS] Union-Find produced {len(valid_tracks)} valid tracks.')

        # 2. Triangulate each track using local-phase optimized poses/depths
        # FIX: get the number of train views to filter test views
        num_cams = self.pose_param_net.num_cams if self.pose_param_net is not None else len(self.local_poses_buffer)
        ## End FIX

        # camera_mat from local_cams_buffer (NeRF normalized coord system)
        if len(self.local_cams_buffer) > 0:
            camera_mat = list(self.local_cams_buffer.values())[0].cpu().numpy()[0]
        else:
            camera_mat = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1]], dtype=np.float32)

        # Compute the scaling factor from depth map to original image
        if len(self.local_depths_buffer) > 0:
            d0 = list(self.local_depths_buffer.values())[0]
            while d0.ndim > 2:
                d0 = d0.squeeze(0)
            h_d, w_d = d0.shape[-2:]
        else:
            h_d, w_d = self.img_h, self.img_w
        scale_x = w_d / float(self.img_w) if self.img_w > 0 else 1.0
        scale_y = h_d / float(self.img_h) if self.img_h > 0 else 1.0
        print(f'[NoPe-NeRF++ TRACKS] Depth map size: {h_d}x{w_d}, Image size: {self.img_h}x{self.img_w}, '
              f'scale: ({scale_x:.4f}, {scale_y:.4f})')
        ## End FIX

        poses = {k: v.detach().cpu().numpy() for k, v in self.local_poses_buffer.items()}
        depths = {k: v.detach().cpu().numpy() for k, v in self.local_depths_buffer.items()}

        track_pts_3d_list = []
        tracks_obs_list = []
        tri_success = 0
        tri_fail = 0

        for obs in valid_tracks:
            # Select first two observations with valid depth for triangulation
            tri_obs = []
            for img_idx, kp_idx, x, y in obs:
                # FIX: skip test views (not in local buffer or index >= num_cams)
                if img_idx not in depths or img_idx >= num_cams:
                    continue
                ## End FIX
                d = depths[img_idx]
                while d.ndim > 2:
                    d = d.squeeze(0)

                # FIX: scale coordinates to depth map size and relax depth threshold
                xi = int(np.clip(x * scale_x, 0, d.shape[1] - 1))
                yi = int(np.clip(y * scale_y, 0, d.shape[0] - 1))
                di = d[yi, xi]
                if np.isfinite(di) and di > 1e-3:
                    tri_obs.append((img_idx, x, y, di))
                ## End FIX
                if len(tri_obs) >= 2:
                    break

            if len(tri_obs) < 2:
                tri_fail += 1
                continue

            img_i, x_i, y_i, di = tri_obs[0]
            img_j, x_j, y_j, dj = tri_obs[1]

            pts_i = np.array([[x_i, y_i]], dtype=np.float32)
            pts_j = np.array([[x_j, y_j]], dtype=np.float32)
            pts_3d, valid_mask = self._triangulate_tracks(
                pts_i, pts_j, poses[img_i], poses[img_j],
                camera_mat, depths[img_i], depths[img_j]
            )
            if valid_mask.sum() == 0:
                tri_fail += 1
                continue

            tri_success += 1
            q = pts_3d[valid_mask][0]  # (3,)
            track_pts_3d_list.append(q)

            # Store all observations in pixel coords for reprojection loss
            # FIX: keep only train-view observations, exclude test views
            obs_list = [(img_idx, float(x), float(y)) for img_idx, kp_idx, x, y in obs if img_idx < num_cams]
            if len(obs_list) < 2:
                tri_fail += 1
                continue
            ## End FIX
            tracks_obs_list.append(obs_list)

        print(f'[NoPe-NeRF++ TRACKS] Triangulation: success={tri_success}, fail={tri_fail}')

        if len(track_pts_3d_list) == 0:
            logger_py.warning('NoPe-NeRF++: No valid tracks after triangulation.')
            self.tracks_extracted_this_run = True
            return

        self.track_pts_3d = nn.Parameter(
            torch.from_numpy(np.stack(track_pts_3d_list, axis=0)).float().to(device)
        )
        self.tracks_observations = tracks_obs_list
        self.tracks_loaded = True
        self.tracks_extracted_this_run = True
        print(f'[NoPe-NeRF++ TRACKS] Created {len(track_pts_3d_list)} track points as Parameters.')
        logger_py.info(f'NoPe-NeRF++: Extracted {len(track_pts_3d_list)} tracks into memory.')

    ## End NoPe-NeRF++ (Sec 4.3)

    ## NoPe-NeRF++ (Sec 4.1/4.2): Helper — get MST edge match coords
    def _get_mst_match_coords(self, i, j):
        i, j = int(i), int(j)
        key = (i, j)
        if key in self.pair_matches:
            mk_i, mk_j, _, _ = self.pair_matches[key]
            return mk_i.copy(), mk_j.copy()
        return None, None

    ## End NoPe-NeRF++

    ## NoPe-NeRF++ (Sec 4.2): Helper — sample random non-adjacent ref from MST
    def _sample_random_ref(self, img_idx):
        if not self.association_ready:
            return None
        img_idx = int(img_idx)
        cands = self.random_match_candidates.get(img_idx, [])
        if len(cands) == 0:
            return None
        neighbors = set()
        for nei, _ in self.mst_adj.get(img_idx, []):
            neighbors.add(nei)
        valid_cands = [c for c in cands if int(c[0]) not in neighbors and int(c[0]) != img_idx]
        if len(valid_cands) == 0:
            return None
        idx = np.random.randint(len(valid_cands))
        return int(valid_cands[idx][0])

    ## End NoPe-NeRF++ (Sec 4.2)

    ## NoPe-NeRF++ (Sec 4.2): Sparse geometric consistency loss
    ## Fully reproduce the coordinate transform chain of transform_to_world in rendering.py:
    ##   p_world = scale_mat^{-1} @ c2w @ camera_mat @ [u_norm, v_norm, d, 1]
    ## ensure exact consistency with the coordinate frame of depth_gt (metric world coordinates),
    ## finally diff * scene_scale normalized to [-1,1] range, MEAN.
    def _compute_sparse_mgc_loss(self, img_idx, ref_idx, mkpts_i, mkpts_j,
                                 depth_i, depth_j, pose_i, pose_j, camera_mat,
                                 scale_mat, device):
        while depth_i.dim() > 2:
            depth_i = depth_i.squeeze(0)
        while depth_j.dim() > 2:
            depth_j = depth_j.squeeze(0)

        if depth_i.dim() != 2 or depth_j.dim() != 2:
            return torch.tensor(0.0, device=device)

        h_dpt, w_dpt = depth_i.shape
        H_img, W_img = self.img_h, self.img_w

        # 1. Scale mkpts to DPT depth resolution, then normalize to [-1, 1]
        sx = w_dpt / float(W_img) if W_img > 0 else 1.0
        sy = h_dpt / float(H_img) if H_img > 0 else 1.0
        pts_i = torch.from_numpy(mkpts_i).float().to(device)
        pts_j = torch.from_numpy(mkpts_j).float().to(device)
        pts_i[:, 0] *= sx; pts_i[:, 1] *= sy
        pts_j[:, 0] *= sx; pts_j[:, 1] *= sy

        # Normalize to [-1, 1] (NeRF pixel convention)
        pts_i[:, 0] = 2.0 * pts_i[:, 0] / (w_dpt - 1) - 1.0
        pts_i[:, 1] = 2.0 * pts_i[:, 1] / (h_dpt - 1) - 1.0
        pts_j[:, 0] = 2.0 * pts_j[:, 0] / (w_dpt - 1) - 1.0
        pts_j[:, 1] = 2.0 * pts_j[:, 1] / (h_dpt - 1) - 1.0

        # 2. Sample depth at scaled keypoint locations
        ix = torch.clamp(((pts_i[:, 0] + 1.0) / 2.0 * (w_dpt - 1)).long(), 0, w_dpt - 1)
        iy = torch.clamp(((pts_i[:, 1] + 1.0) / 2.0 * (h_dpt - 1)).long(), 0, h_dpt - 1)
        jx = torch.clamp(((pts_j[:, 0] + 1.0) / 2.0 * (w_dpt - 1)).long(), 0, w_dpt - 1)
        jy = torch.clamp(((pts_j[:, 1] + 1.0) / 2.0 * (h_dpt - 1)).long(), 0, h_dpt - 1)

        di = depth_i[iy, ix]
        dj = depth_j[jy, jx]

        valid = (di > 1e-3) & (dj > 1e-3) & torch.isfinite(di) & torch.isfinite(dj)
        if valid.sum().item() == 0:
            return torch.tensor(0.0, device=device)

        di, dj = di[valid], dj[valid]
        ui, vi = pts_i[valid, 0], pts_i[valid, 1]
        uj, vj = pts_j[valid, 0], pts_j[valid, 1]

        # 3. Reproduce transform_to_world logic step-by-step
        # camera_mat @ [u, v, d, 1] = [fx*u, fy*v, 1, d]
        _cm = camera_mat.squeeze(0) if camera_mat.dim() == 3 else camera_mat
        fx_n = abs(float(_cm[0, 0]))
        fy_n = abs(float(_cm[1, 1]))
        Xi = torch.stack([fx_n * ui, fy_n * vi, torch.ones_like(di), di], dim=-1)  # (N,4)
        Xj = torch.stack([fx_n * uj, fy_n * vj, torch.ones_like(dj), dj], dim=-1)  # (N,4)

        # c2w @ [fx*u, fy*v, 1, d] = R @ [fx*u, fy*v, 1] + t * d
        Pi = (pose_i @ Xi.T).transpose(0, 1)[:, :3]  # (N,3), normalized world coords * d
        Pj = (pose_j @ Xj.T).transpose(0, 1)[:, :3]  # (N,3)

        # scale_mat^{-1} @ Pi = meter-scale world coords
        # (same as transform_to_world output)
        _s = scale_mat.squeeze(0) if scale_mat.dim() == 3 else scale_mat  # (4,4)
        s_inv = torch.inverse(_s)  # (4,4)
        Pi_hom = torch.cat([Pi, torch.ones((Pi.shape[0], 1), device=device)], dim=1)  # (N,4)
        Pj_hom = torch.cat([Pj, torch.ones((Pj.shape[0], 1), device=device)], dim=1)  # (N,4)
        Wi = (s_inv @ Pi_hom.T).transpose(0, 1)[:, :3]  # (N,3), meters
        Wj = (s_inv @ Pj_hom.T).transpose(0, 1)[:, :3]  # (N,3), meters

        # 4. Eq.4: compute diff in meter-scale world coords
        diff = Wi - Wj  # (N,3), meters

        # 5. Normalize: diff * scene_scale → [-1, 1] range
        # scene_scale = Frobenius(scale_mat[:3,:3]) ≈ sqrt(3)/s for diag scale
        scene_scale = torch.norm(_s[:3, :3], p='fro', dim=(-2, -1)).mean().item()
        scene_scale = max(scene_scale, 1e-6)
        diff_norm = diff * scene_scale  # meters * (1/s) = normalized coords

        # MEAN (not SUM) so loss magnitude is independent of match count
        loss = torch.sqrt((diff_norm ** 2).sum(dim=-1) + 1e-6).mean()
        return loss
    def _compute_track_reprojection_loss(self, camera_mat_gt, device):
        if self.track_pts_3d is None or not self.tracks_loaded:
            return torch.tensor(0.0, device=device)

        # FIX: get the number of train views to filter test view indices
        num_cams = self.pose_param_net.num_cams if self.pose_param_net is not None else 0
        ## End FIX

        M = self.track_pts_3d.shape[0]
        n_sample = min(self.n_track_samples, M)
        sample_idx = torch.randperm(M, device=device)[:n_sample]

        obs_img_idx = []
        obs_uv_gt = []
        obs_track_idx = []

        for idx in sample_idx:
            idx_int = int(idx.item())
            if idx_int >= len(self.tracks_observations):
                continue
            for obs in self.tracks_observations[idx_int]:
                # FIX: keep only train-view observations, exclude test views
                if obs[0] < num_cams:
                    obs_img_idx.append(obs[0])
                    obs_uv_gt.append([obs[1], obs[2]])
                    obs_track_idx.append(idx_int)
                ## End FIX

        if len(obs_img_idx) == 0:
            return torch.tensor(0.0, device=device)

        obs_img_idx_t = torch.tensor(obs_img_idx, device=device, dtype=torch.long)
        obs_uv_gt_t = torch.tensor(obs_uv_gt, device=device, dtype=torch.float32)  # (O,2) pixel coords
        obs_track_idx_t = torch.tensor(obs_track_idx, device=device, dtype=torch.long)

        q_batch = self.track_pts_3d[obs_track_idx_t]  # (O, 3)

        # Gather poses for all observed views
        unique_img_idx = torch.unique(obs_img_idx_t)
        c2w_dict = {}
        for uidx in unique_img_idx:
            uidx_int = int(uidx.item())
            c2w_dict[uidx_int] = self.pose_param_net(torch.tensor([uidx_int], device=device))

        c2w_batch = torch.stack([c2w_dict[int(i.item())] for i in obs_img_idx_t])  # (O,4,4)
        w2c_batch = torch.inverse(c2w_batch)  # (O,4,4)

        q_homo = torch.cat([q_batch, torch.ones(len(obs_img_idx), 1, device=device)], dim=-1)  # (O,4)
        q_cam = torch.bmm(w2c_batch, q_homo.unsqueeze(-1)).squeeze(-1)[:, :3]  # (O,3)

        # NeRF camera_mat projection (cx=cy=0, outputs normalized coords [-1,1])
        _cm = camera_mat_gt.squeeze(0) if camera_mat_gt.dim() == 3 else camera_mat_gt
        fx_n = abs(float(_cm[0, 0]))
        fy_n = abs(float(_cm[1, 1]))

        z_cam = q_cam[:, 2] + 1e-6
        u_pred = fx_n * q_cam[:, 0] / z_cam  # [-1, 1]
        v_pred = fy_n * q_cam[:, 1] / z_cam  # [-1, 1]
        uv_pred_pixel = torch.stack([u_pred, v_pred], dim=-1)  # (O,2)

        # FIX: camera_mat_gt uses normalized coords [-1,1], so obs_uv_gt_t must
        # be normalized from pixel coords [0,W-1] to [-1,1] before computing diff.
        obs_uv_gt_norm = obs_uv_gt_t.clone()
        obs_uv_gt_norm[:, 0] = 2.0 * obs_uv_gt_norm[:, 0] / float(self.img_w - 1) - 1.0
        obs_uv_gt_norm[:, 1] = 2.0 * obs_uv_gt_norm[:, 1] / float(self.img_h - 1) - 1.0

        diff = uv_pred_pixel - obs_uv_gt_norm
        # Paper Eq.7: L_track (pixel-level reprojection error)
        loss_proj = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-6).mean()
        return loss_proj
        ## End FIX

    ## End NoPe-NeRF++ (Sec 4.3)

    ## NoPe-NeRF++ (Sec 4.3): Triangulate matched keypoints (numpy, no grad)
    ## Fully reproduce the coordinate transform chain of transform_to_world (Steps 1-3),
    ## returning normalized world coordinates (consistent with the NeRF coordinate frame).
    def _triangulate_tracks(self, mkpts_i, mkpts_j, pose_i, pose_j,
                            camera_mat, depth_i, depth_j):
        if isinstance(depth_i, torch.Tensor):
            depth_i = depth_i.detach().cpu().numpy()
        if isinstance(depth_j, torch.Tensor):
            depth_j = depth_j.detach().cpu().numpy()

        while depth_i.ndim > 2:
            depth_i = depth_i.squeeze(0)
        while depth_j.ndim > 2:
            depth_j = depth_j.squeeze(0)

        if depth_i.ndim != 2 or depth_j.ndim != 2:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=bool)

        h_dpt, w_dpt = depth_i.shape
        H_img, W_img = self.img_h, self.img_w

        # 1. Scale mkpts to DPT resolution, normalize to [-1, 1]
        sx = w_dpt / float(W_img) if W_img > 0 else 1.0
        sy = h_dpt / float(H_img) if H_img > 0 else 1.0
        pts_i = mkpts_i.copy().astype(np.float32)
        pts_j = mkpts_j.copy().astype(np.float32)
        pts_i[:, 0] *= sx; pts_i[:, 1] *= sy
        pts_j[:, 0] *= sx; pts_j[:, 1] *= sy
        pts_i[:, 0] = 2.0 * pts_i[:, 0] / (w_dpt - 1) - 1.0
        pts_i[:, 1] = 2.0 * pts_i[:, 1] / (h_dpt - 1) - 1.0
        pts_j[:, 0] = 2.0 * pts_j[:, 0] / (w_dpt - 1) - 1.0
        pts_j[:, 1] = 2.0 * pts_j[:, 1] / (h_dpt - 1) - 1.0

        # Sample depth
        ix = np.clip(((pts_i[:, 0] + 1.0) / 2.0 * (w_dpt - 1)).astype(np.int32), 0, w_dpt - 1)
        iy = np.clip(((pts_i[:, 1] + 1.0) / 2.0 * (h_dpt - 1)).astype(np.int32), 0, h_dpt - 1)
        jx = np.clip(((pts_j[:, 0] + 1.0) / 2.0 * (w_dpt - 1)).astype(np.int32), 0, w_dpt - 1)
        jy = np.clip(((pts_j[:, 1] + 1.0) / 2.0 * (h_dpt - 1)).astype(np.int32), 0, h_dpt - 1)

        d_i = depth_i[iy, ix]
        d_j = depth_j[jy, jx]

        valid = (d_i > 1e-3) & (d_j > 1e-3) & np.isfinite(d_i) & np.isfinite(d_j)
        if valid.sum() == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=bool)

        # 2. camera_mat @ [u, v, d, 1] = [fx*u, fy*v, 1, d]
        # Squeeze batch dim if present, then 2D index
        _cm = camera_mat.squeeze() if camera_mat.ndim == 3 else camera_mat
        fx_n = abs(float(_cm[0, 0]))
        fy_n = abs(float(_cm[1, 1]))
        di_v, dj_v = d_i[valid], d_j[valid]
        ui, vi = pts_i[valid, 0], pts_i[valid, 1]
        uj, vj = pts_j[valid, 0], pts_j[valid, 1]

        Xi = np.stack([fx_n * ui, fy_n * vi, np.ones_like(di_v), di_v], axis=-1)  # (N,4)
        Xj = np.stack([fx_n * uj, fy_n * vj, np.ones_like(dj_v), dj_v], axis=-1)  # (N,4)

        # 3. c2w @ [fx*u, fy*v, 1, d] = R @ [...] + t*d
        # This gives NORMALIZED world coords (same as Step 3 in transform_to_world)
        Pi = (pose_i @ Xi.T).T[:, :3]  # (N,3), normalized world coords
        Pj = (pose_j @ Xj.T).T[:, :3]

        # Average for triangulation
        pts_3d_norm = (Pi + Pj) / 2.0  # (N,3), normalized world coords

        full_3d = np.zeros((len(mkpts_i), 3), dtype=np.float32)
        full_valid = np.zeros(len(mkpts_i), dtype=bool)
        full_3d[valid] = pts_3d_norm
        full_valid[valid] = True
        return full_3d, full_valid
    ## End NoPe-NeRF++ (Sec 4.3)

    ## NoPe-NeRF++ (Sec 4.1): OpenCV SIFT feature extraction and matching (utility / fallback)
    def _extract_match_cv(self, img_path_i, img_path_j):
        self._init_cv_sift()
        img_i = cv2.imread(img_path_i, cv2.IMREAD_GRAYSCALE)
        img_j = cv2.imread(img_path_j, cv2.IMREAD_GRAYSCALE)
        if img_i is None or img_j is None:
            return None, None, None
        kp_i, des_i = self._cv_sift.detectAndCompute(img_i, None)
        kp_j, des_j = self._cv_sift.detectAndCompute(img_j, None)
        if des_i is None or des_j is None or len(kp_i) == 0 or len(kp_j) == 0:
            return None, None, None
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des_i, des_j, k=2)
        good = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n[0], m_n[1]
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) == 0:
            return None, None, None
        mkpts_i = np.float32([kp_i[m.queryIdx].pt for m in good])
        mkpts_j = np.float32([kp_j[m.trainIdx].pt for m in good])
        scores = np.float32([1.0 / (1.0 + m.distance) for m in good])
        return mkpts_i, mkpts_j, scores

    ## End NoPe-NeRF++ (Sec 4.1)

    ## NoPe-NeRF++ (Sec 4.3): Load pre-computed tracks from disk (optional, legacy)
    def _load_tracks_from_disk(self):
        if self.tracks_loaded:
            return
        if not os.path.exists(self.tracks_dir):
            return
        files = [f for f in os.listdir(self.tracks_dir) if f.startswith('tracks_') and f.endswith('.pt')]
        if len(files) == 0:
            return
        for f in files:
            parts = f.replace('tracks_', '').replace('.pt', '').split('_')
            i, j = int(parts[0]), int(parts[1])
            tracks = torch.load(os.path.join(self.tracks_dir, f), map_location='cpu')
            self.tracks_cache[(i, j)] = tracks
            rev = {
                'pts_3d': tracks['pts_3d'],
                'pts_2d_i': tracks['pts_2d_j'],
                'pts_2d_j': tracks['pts_2d_i'],
                'track_ids': tracks['track_ids'],
                'visibility': tracks['visibility'],
            }
            self.tracks_cache[(j, i)] = rev
        self.tracks_loaded = True
        logger_py.info(f'NoPe-NeRF++: Loaded {len(files)} track pairs from {self.tracks_dir}')

    ## End NoPe-NeRF++ (Sec 4.3)

    def train_step(self, data, it=None, epoch=None, scheduling_start=None, render_path=None):
        """ Performs a training step.
        """
        # FIX: unify the local variable device to avoid NameError in the MST MGC / Random Matching blocks below
        device = self.device
        ## End FIX

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

        img_idx_raw = data.get('img.idx', -1)
        img_idx = int(img_idx_raw.item()) if isinstance(img_idx_raw, torch.Tensor) else int(img_idx_raw)

        ## NoPe-NeRF++ (Sec 4.2/4.3): Delayed collection for resume
        if self.delayed_collection and img_idx >= 0:
            self.local_poses_buffer[img_idx] = self.pose_param_net(img_idx).detach().clone()
            dpt = data.get('img.dpt', None)
            if dpt is not None:
                # FIX: ensure dpt is on GPU, consistent with scale_i / shift_i device
                dpt = dpt.to(self.device)
                ## End FIX
                if self.distortion_net is not None:
                    scale_i, shift_i = self.distortion_net(torch.tensor([img_idx], device=self.device))
                    scale_i = scale_i.to(self.device)
                    shift_i = shift_i.to(self.device)
                    if self.shift_first:
                        dpt_store = (dpt + shift_i) * scale_i
                    else:
                        dpt_store = dpt * scale_i + shift_i
                else:
                    dpt_store = dpt
                self.local_depths_buffer[img_idx] = dpt_store.detach().clone()
            cam = data.get('img.camera_mat', None)
            if cam is not None:
                # FIX: sync camera_mat device
                cam = cam.to(self.device)
                ## End FIX
                self.local_cams_buffer[img_idx] = cam.detach().clone()

            target_views = self.pose_param_net.num_cams if self.pose_param_net is not None else len(
                self.local_poses_buffer)
            if len(self.local_poses_buffer) >= target_views:
                print(f'=> NoPe-NeRF++: Delayed collection complete. Extracting tracks...')
                self._extract_global_tracks(self.device)
                self.delayed_collection = False
        ## End NoPe-NeRF++ (Sec 4.2/4.3)

        ## NoPe-NeRF++ (Sec 4.2): Auto-collect local phase data + frame cache for random matching
        # FIX: pose buffer is updated in all phases (not just the local phase)
        # so eval can see the latest pose during the global phase
        if it is not None and img_idx >= 0:
            self.local_poses_buffer[img_idx] = self.pose_param_net(img_idx).detach().clone()
        if it is not None and it <= self.local_phase_steps:
            if img_idx >= 0:
                dpt = data.get('img.dpt', None)
                if dpt is not None:
                    # FIX: ensure dpt is on GPU
                    dpt = dpt.to(self.device)
                    ## End FIX
                    # Store transformed depth D* for track triangulation
                    if self.distortion_net is not None:
                        scale_i, shift_i = self.distortion_net(torch.tensor([img_idx], device=self.device))
                        scale_i = scale_i.to(self.device)
                        shift_i = shift_i.to(self.device)
                        if self.shift_first:
                            dpt_store = (dpt + shift_i) * scale_i
                        else:
                            dpt_store = dpt * scale_i + shift_i
                    else:
                        dpt_store = dpt
                    self.local_depths_buffer[img_idx] = dpt_store.detach().clone()
                cam = data.get('img.camera_mat', None)
                if cam is not None:
                    # FIX: sync camera_mat device
                    cam = cam.to(self.device)
                    ## End FIX
                    self.local_cams_buffer[img_idx] = cam.detach().clone()

                # Cache raw frames for random matching scheme
                img_raw = data.get('img')
                dpt_raw = data.get('img.dpt')
                if img_raw is not None:
                    self.frame_img_cache[int(img_idx)] = img_raw.detach().cpu()
                if dpt_raw is not None:
                    self.frame_dpt_cache[int(img_idx)] = dpt_raw.detach().cpu()

                if self.pose_param_net is not None and len(self.frame_img_cache) >= self.pose_param_net.num_cams:
                    self.frame_cache_full = True

            ## NoPe-NeRF++ (Sec 4.2): after all views are collected for the first time, perform ICP initialization
            if (self.pose_param_net is not None and
                    len(self.local_depths_buffer) >= self.pose_param_net.num_cams and
                    not self.icp_init_done):
                print(f'=> NoPe-NeRF++: All {self.pose_param_net.num_cams} views buffered. Running ICP init...')
                self._icp_init_poses(self.device)
            ## End NoPe-NeRF++ (Sec 4.2)

            if it > 0 and it % 10000 == 0 and not SP_SG_AVAILABLE and not CV2_AVAILABLE:
                logger_py.warning(
                    'NoPe-NeRF++: Neither SuperPoint+SuperGlue nor OpenCV available. '
                    'Track extraction will fail at step {}. '.format(self.local_phase_steps)
                )
        ## End NoPe-NeRF++ (Sec 4.2)

        loss_dict = self.compute_loss(data, it=it, epoch=epoch, scheduling_start=scheduling_start,
                                      out_render_path=render_path)
        loss = loss_dict['loss']
        loss.backward()

        ## NoPe-NeRF++ (Sec 5.1): Psi is optimized only in the local phase
        if it is not None and self.distortion_net is not None and it >= self.local_phase_steps:
            for p in self.distortion_net.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        ## End NoPe-NeRF++ (Sec 5.1)

        self.optimizer.step()
        if self.optimizer_pose:
            self.optimizer_pose.step()
        if self.optimizer_focal:
            self.optimizer_focal.step()


        if self.optimizer_distortion:
            if it is None or it < self.local_phase_steps:
                self.optimizer_distortion.step()
        ## End NoPe-NeRF++ (Sec 5.1)

        ## NoPe-NeRF++ (Sec 4.3): Trigger global track extraction once at local phase end
        if it is not None and it >= self.local_phase_steps and not self.delayed_collection:
            # FIX: even if tracks_loaded=True (e.g. old cache loaded from disk), if track_pts_3d is None it must be re-extracted
            need_extract = (self.track_pts_3d is None) or (not self.tracks_loaded)
            if need_extract:
                if not self.tracks_extracted_this_run:
                    if len(self.local_poses_buffer) > 0:
                        print(f'=> NoPe-NeRF++: it={it}, extracting global tracks...')
                        self._extract_global_tracks(self.device)
                    else:
                        print(f'=> NoPe-NeRF++: it={it}, buffer empty. Starting delayed collection...')
                        self.delayed_collection = True
                        self.tracks_extracted_this_run = True
                elif it % 5000 == 0:
                    logger_py.warning(
                        f'[NoPe-NeRF++] Track extraction previously failed. L_track remains disabled.'
                    )
        ## End NoPe-NeRF++ (Sec 4.3)

        ## NoPe-NeRF++ (Sec 4.3): Optimize track points Q (manual SGD, since Q is not in external optimizer)
        if it is not None and it >= self.local_phase_steps and self.track_pts_3d is not None:
            if self.track_pts_3d.grad is not None:
                with torch.no_grad():
                    self.track_pts_3d -= self.track_lr * self.track_pts_3d.grad
                self.track_pts_3d.grad.zero_()
        ## End NoPe-NeRF++ (Sec 4.3)


        return loss_dict


    ## NoPe-NeRF++: Export current poses for eval
    def export_poses(self):
        """Return dict of current poses from pose_param_net (latest, not cached)."""
        if self.pose_param_net is None:
            return {}
        poses = {}
        with torch.no_grad():
            for idx in range(self.pose_param_net.num_cams):
                c2w = self.pose_param_net(idx).detach().cpu()
                poses[idx] = c2w
        return poses
    ## End NoPe-NeRF++
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
            depth_pred_out = depth_pred.view(h, w).detach().cpu().numpy()
            imageio.imwrite(os.path.join(out_render_path, '%04d_depth.png' % img_idx),
                            np.clip(255.0 / depth_pred_out.max() * (depth_pred_out - depth_pred_out.min()), 0,
                                    255).astype(np.uint8))

            img1 = Image.fromarray(
                (img_out).astype(np.uint8)
            ).convert("RGB").save(
                os.path.join(out_render_path, '%04d_img.png' % img_idx)
            )
        if self.vis_geo:
            with torch.no_grad():
                rgb_pred = [self.model(
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

    def compute_loss(self, data, eval_mode=False, it=None, epoch=None, scheduling_start=None, out_render_path=None):
        weights = {}
        ## NoPe-NeRF++ (Sec 5.1): keep the original NoPe-NeRF annealing logic unchanged
        weights_name_list = ['rgb_weight', 'depth_weight', 'pc_weight', 'rgb_s_weight', 'depth_consistency_weight',
                             'weight_dist_2nd_loss', 'weight_dist_1st_loss']
        weights_list = [
            self.anneal(getattr(self, w)[0], getattr(self, w)[1], scheduling_start, self.annealing_epochs, epoch) for w
            in weights_name_list]
        rgb_loss_type = 'l1' if epoch < self.annealing_epochs + scheduling_start else 'l2'

        for i, weight in enumerate(weights_list):
            weight_name = weights_name_list[i]
            weights[weight_name] = weight
        ## End NoPe-NeRF++ (Sec 5.1)

        ## NoPe-NeRF++ (Sec 5.1): phase-based weight override - strictly following the paper
        if it is not None:
            if it < self.local_phase_steps:
                # Local phase: L_local = L_rgb + lambda_depth L_depth + lambda_mgc L_mgc
                weights['depth_weight'] = self.lambda_depth_local
                # Disable NoPe-NeRF original dense chamfer distance / rgb_s / depth_consistency
                weights['pc_weight'] = 0.0
                weights['rgb_s_weight'] = 0.0
                weights['depth_consistency_weight'] = 0.0
                weights['feature_weight'] = 0.0
                # FIX-1: disable weight_dist loss (not part of the NoPe-NeRF++ paper)
                weights['weight_dist_1st_loss'] = 0.0
                weights['weight_dist_2nd_loss'] = 0.0
            else:
                global_it = it - self.local_phase_steps
                if global_it < self.global_phase_steps_first:
                    track_w = self.lambda_track_global_first
                else:
                    track_w = self.lambda_track_global_second

                weights['feature_weight'] = track_w
                # Global phase disables all local dense losses
                weights['pc_weight'] = 0.0
                weights['rgb_s_weight'] = 0.0
                weights['depth_consistency_weight'] = 0.0
                weights['depth_weight'] = 0.0
                # FIX-1: disable weight_dist loss (not part of the NoPe-NeRF++ paper)
                weights['weight_dist_1st_loss'] = 0.0
                weights['weight_dist_2nd_loss'] = 0.0
        ## End NoPe-NeRF++ (Sec 5.1) + FIX-1

        # FIX-2: ensure rgb_weight is always 1.0 in the local phase (photometric supervision is required)
        if it is not None and it < self.local_phase_steps:
            weights['rgb_weight'] = 1.0

        ## NoPe-NeRF++ (Sec 5.1): print current phase status log every 1000 steps
        if it is not None and it % 1000 == 0:
            if it < self.local_phase_steps:
                logger_py.info(
                    f'[NoPe-NeRF++ LOCAL PHASE] it={it}/{self.local_phase_steps} | '
                    f'lambda_depth={weights.get("depth_weight", 0):.4f} '
                    f'lambda_mgc(sparse)={self.lambda_mgc_local:.2f} '
                    f'Psi_grad={"ON" if (self.distortion_net and any(p.grad is not None for p in self.distortion_net.parameters())) else "OFF"}'
                )
            else:
                global_it = it - self.local_phase_steps
                phase_name = 'GLOBAL FIRST' if global_it < self.global_phase_steps_first else 'GLOBAL SECOND'
                logger_py.info(
                    f'[NoPe-NeRF++ {phase_name}] it={it} (global={global_it}) | '
                    f'lambda_track={weights.get("feature_weight", 0):.2f} | '
                    f'Psi_grad=OFF | '
                    f'tracks_loaded={self.tracks_loaded} | '
                    f'N_tracks={self.track_pts_3d.shape[0] if self.track_pts_3d is not None else 0}'
                )
        ## End NoPe-NeRF++ (Sec 5.1)

        render_model = (weights['rgb_weight'] != 0.0) or (weights['depth_weight'] != 0.0)
        ## NoPe-NeRF++: L_track does NOT need ref_imgs/point clouds.
        ## Only enable when old NoPe-NeRF dense losses are active.
        use_ref_imgs = ((weights['pc_weight'] != 0.0) or (weights['rgb_s_weight'] != 0.0) or
                        (weights['depth_consistency_weight'] != 0.0))

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
            scale_input = scale_input.to(self.device)
            shift_input = shift_input.to(self.device)
            if self.shift_first:
                depth_input = (depth_input + shift_input) * scale_input
            else:
                depth_input = depth_input * scale_input + shift_input

        if self.optimizer_focal:
            fxfy = self.focal_net(0)
            pad = torch.zeros(4).to(device)
            one = torch.tensor([1]).to(device)
            camera_mat = torch.cat([fxfy[0:1], pad, -fxfy[1:2], pad, -one, pad, one])
            camera_mat = camera_mat.view(1, 4, 4)
        else:
            camera_mat = camera_mat_gt

        # Sample pixels
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

        if use_ref_imgs:
            c2w_ref = self.pose_param_net(ref_idx)
            if self.distortion_net is not None:
                scale_ref, shift_ref = self.distortion_net(ref_idx)
                scale_ref = scale_ref.to(self.device)
                shift_ref = shift_ref.to(self.device)
                if self.shift_first:
                    depth_ref = scale_ref * (depth_ref + shift_ref)
                else:
                    depth_ref = scale_ref * depth_ref + shift_ref
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

            d1 = torch.clamp(d1, min=nl)
            d2 = torch.clamp(d2, min=nl)

            pc1 = transform_to_world(p_pc, d1.view(1, -1, 1), camera_mat)
            pc2 = transform_to_world(p_pc, d2.view(1, -1, 1), camera_mat)

            if weights['depth_consistency_weight'] != 0.0:
                B, N, _ = pc1.shape
                pc1_homo = torch.cat([pc1, torch.ones(B, N, 1, device=device)], dim=-1).permute(0, 2, 1)
                pc1_cam2_homo = Rt_rel_12 @ pc1_homo
                pc1_cam2 = pc1_cam2_homo[:, :3, :].permute(0, 2, 1)
                d1_proj = -pc1_cam2[..., 2]

                pc2_homo = torch.cat([pc2, torch.ones(B, N, 1, device=device)], dim=-1).permute(0, 2, 1)
                pc2_cam1_homo = torch.inverse(Rt_rel_12) @ pc2_homo
                pc2_cam1 = pc2_cam1_homo[:, :3, :].permute(0, 2, 1)
                d2_proj = -pc2_cam1[..., 2]

                kwargs['d1_proj'] = d1_proj
                kwargs['d2'] = d2.view(B, -1)
                kwargs['d2_proj'] = d2_proj
                kwargs['d1'] = d1.view(B, -1)
            else:
                kwargs['d1_proj'] = None
                kwargs['d2'] = None
                kwargs['d2_proj'] = None
                kwargs['d1'] = None

            if weights['rgb_s_weight'] != 0.0:
                img1 = F.interpolate(img1, sample_resolution, mode='bilinear')
                img2 = F.interpolate(img2, sample_resolution, mode='bilinear')

                if weights['rgb_s_weight'] != 0.0:
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

        ## NoPe-NeRF++ (Sec 4.3): Global Track Constraint L_track
        ## Moved outside use_ref_imgs: L_track only needs camera_mat_gt & poses,
        ## does NOT require ref_imgs or point clouds.
        if weights.get('feature_weight', 0.0) != 0.0:
            if self.track_pts_3d is not None and self.tracks_loaded:
                feature_loss = self._compute_track_reprojection_loss(camera_mat_gt, device)
                if it is not None and it % 500 == 0:
                    logger_py.info(
                        f'[NoPe-NeRF++ L_TRACK ACTIVE] it={it} | '
                        f'N_tracks={self.track_pts_3d.shape[0]} | '
                        f'L_track_mean={feature_loss.item():.6f} | '
                        f'lambda_track={weights["feature_weight"]:.2f}'
                    )
            else:
                feature_loss = torch.tensor(0.0, device=device)
                if it is not None and it % 2000 == 0:
                    logger_py.warning(
                        f'[NoPe-NeRF++ L_TRACK=0] it={it} | '
                        f'tracks_loaded={self.tracks_loaded} | '
                        f'track_pts_3d exists={self.track_pts_3d is not None}'
                    )
            kwargs['feature_loss'] = feature_loss
        ## End NoPe-NeRF++ (Sec 4.3)

        if render_model and self.detach_gt_depth:
            gt_depth = gt_depth.detach()
        loss_dict = self.loss(rendered_rgb, rgb_gt, rendered_depth, gt_depth, **kwargs)

        ## NoPe-NeRF++ (Sec 4.2): MST-based sparse geometric consistency (parent & children)
        if it is not None and it < self.local_phase_steps and self.association_ready:
            int_img_idx = int(img_idx.item()) if isinstance(img_idx, torch.Tensor) else int(img_idx)
            loss_mst_mgc = torch.tensor(0.0, device=device)
            count_mst = 0

            def _get_cached_depth_transformed(idx):
                dpt_raw = self.frame_dpt_cache.get(idx)
                if dpt_raw is None:
                    return None
                dpt = dpt_raw.to(device)
                if dpt.dim() == 2:
                    dpt = dpt.unsqueeze(0).unsqueeze(0)
                elif dpt.dim() == 3:
                    dpt = dpt.unsqueeze(1)
                if self.distortion_net is not None:
                    scale, shift = self.distortion_net(torch.tensor([idx], device=device))
                    scale = scale.to(device)
                    shift = shift.to(device)
                    if self.shift_first:
                        dpt = (dpt + shift) * scale
                    else:
                        dpt = dpt * scale + shift
                return dpt

            # Parent edge
            p_idx = self.mst_parent.get(int_img_idx)
            if p_idx is not None:
                mkpts_i, mkpts_p = self._get_mst_match_coords(int_img_idx, p_idx)
                d_p = _get_cached_depth_transformed(p_idx)
                if mkpts_i is not None and len(mkpts_i) > 0 and d_p is not None:
                    pose_p = self.pose_param_net(torch.tensor([p_idx], device=device))
                    loss_p = self._compute_sparse_mgc_loss(
                        int_img_idx, p_idx, mkpts_i, mkpts_p,
                        depth_input, d_p, c2w, pose_p, camera_mat_gt,
                        scale_mat, device
                    )
                    loss_mst_mgc = loss_mst_mgc + loss_p
                    count_mst += 1

            # Children edges
            for c_idx in self.mst_children.get(int_img_idx, []):
                mkpts_i, mkpts_c = self._get_mst_match_coords(int_img_idx, c_idx)
                d_c = _get_cached_depth_transformed(c_idx)
                if mkpts_i is not None and len(mkpts_i) > 0 and d_c is not None:
                    pose_c = self.pose_param_net(torch.tensor([c_idx], device=device))
                    loss_c = self._compute_sparse_mgc_loss(
                        int_img_idx, c_idx, mkpts_i, mkpts_c,
                        depth_input, d_c, c2w, pose_c, camera_mat_gt,
                        scale_mat, device
                    )
                    loss_mst_mgc = loss_mst_mgc + loss_c
                    count_mst += 1

            if count_mst > 0:
                # Paper Eq.6 sums over parent/child/random
                loss_dict['loss'] = loss_dict['loss'] + self.lambda_mgc_local * loss_mst_mgc
                loss_dict['loss_mst_mgc'] = loss_mst_mgc
                if it is not None and it % 1000 == 0:
                    logger_py.info(
                        f'[NoPe-NeRF++ MST MGC] it={it} | img={int_img_idx} | '
                        f'parent={p_idx} | n_children={len(self.mst_children.get(int_img_idx, []))} | '
                        f'count={count_mst} | L_mst_mgc={loss_mst_mgc.item():.6f}'
                    )
        ## End NoPe-NeRF++ (Sec 4.2)

        ## NoPe-NeRF++ (Sec 4.2): Random Matching Scheme (sparse MGC on non-adjacent MST ref)
        if it is not None and it < self.local_phase_steps and self.frame_cache_full and self.association_ready:
            int_img_idx = int(img_idx.item()) if isinstance(img_idx, torch.Tensor) else int(img_idx)
            rand_idx = self._sample_random_ref(int_img_idx)
            if rand_idx is not None and rand_idx in self.frame_dpt_cache:
                rand_dpt_raw = self.frame_dpt_cache[rand_idx].to(device)
                if rand_dpt_raw.dim() == 2:
                    rand_dpt_raw = rand_dpt_raw.unsqueeze(0).unsqueeze(0)
                elif rand_dpt_raw.dim() == 3:
                    rand_dpt_raw = rand_dpt_raw.unsqueeze(1)

                if self.distortion_net is not None:
                    scale_r, shift_r = self.distortion_net(torch.tensor([rand_idx], device=device))
                    if self.shift_first:
                        rand_dpt = (rand_dpt_raw + shift_r) * scale_r
                    else:
                        rand_dpt = rand_dpt_raw * scale_r + shift_r
                else:
                    rand_dpt = rand_dpt_raw

                rand_pose = self.pose_param_net(torch.tensor([rand_idx], device=device))
                mkpts_i, mkpts_j = self._get_mst_match_coords(int_img_idx, rand_idx)
                if mkpts_i is not None and len(mkpts_i) > 0:
                    rand_mgc_loss = self._compute_sparse_mgc_loss(
                        int_img_idx, rand_idx, mkpts_i, mkpts_j,
                        depth_input, rand_dpt, c2w, rand_pose, camera_mat_gt,
                        scale_mat, device
                    )
                    loss_dict['loss'] = loss_dict['loss'] + self.lambda_mgc_local * rand_mgc_loss
                    loss_dict['loss_rand_mgc'] = rand_mgc_loss
                    if it is not None and it % 1000 == 0:
                        logger_py.info(
                            f'[NoPe-NeRF++ RANDOM MGC] it={it} | '
                            f'img={int_img_idx} | rand_ref={rand_idx} | '
                            f'N_matches={len(mkpts_i)} | L_rand_mgc={rand_mgc_loss.item():.6f}'
                        )
        ## End NoPe-NeRF++ (Sec 4.2)

        ## NoPe-NeRF++ (Sec 4.3): Aggregate Global Track Loss
        if weights.get('feature_weight', 0.0) != 0.0 and 'feature_loss' in kwargs:
            loss_dict['loss'] = loss_dict['loss'] + weights['feature_weight'] * kwargs['feature_loss']
            loss_dict['loss_feature'] = kwargs['feature_loss']
        ## End NoPe-NeRF++ (Sec 4.3)


        if self.optimizer_focal:
            _cm = camera_mat_gt.squeeze(0) if camera_mat_gt.dim() == 3 else camera_mat_gt
            loss_dict['focalx'] = fxfy[0] / float(_cm[0, 0])
            loss_dict['focaly'] = fxfy[1] / float(_cm[1, 1])

        loss_dict['scale'] = scale_input
        loss_dict['shift'] = shift_input
        return loss_dict