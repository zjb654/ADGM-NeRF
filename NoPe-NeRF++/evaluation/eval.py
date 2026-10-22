import os
from re import L
import sys
import argparse
import time
import logging
import torch

sys.path.append(os.path.join(sys.path[0], '..'))
from dataloading import get_dataloader, load_config
from model.checkpoints import CheckpointIO
from model.common import compute_errors
from model.eval_images import Eval_Images
import model as mdl
import imageio
import numpy as np
import lpips as lpips_lib
from utils_poses.align_traj import align_scale_c2b_use_a2b, align_ate_c2b_use_a2b
from tqdm import tqdm
from model.common import mse2psnr
from torch.utils.tensorboard import SummaryWriter

# [NEW] Import matplotlib for plotting
import matplotlib.pyplot as plt


def analyze_and_plot_depth(depth_preds_tensor, output_dir, min_depth=1e-6, max_depth=100.0):
    """
    Analyze depth data, generate histogram and curve plots, and print statistics.
    Accepts min_depth and max_depth to limit the statistics range.
    """
    print(f"Starting depth analysis with range [{min_depth}, {max_depth}]...")
    depth_flat = depth_preds_tensor.numpy().flatten()

    valid_mask = (depth_flat >= min_depth) & (depth_flat <= max_depth) & (~np.isnan(depth_flat))
    depth_valid = depth_flat[valid_mask]

    if depth_valid.size == 0:
        print("No valid depth data found for analysis.")
        return

    count_low = np.sum((depth_valid >= min_depth) & (depth_valid < 1))
    count_mid = np.sum((depth_valid >= 1) & (depth_valid < 10))
    count_high = np.sum((depth_valid >= 10) & (depth_valid <= max_depth))

    total_valid = depth_valid.size

    stats_text = (
        f"Depth Statistics:\n"
        f"Total pixels in range [{min_depth}, {max_depth}]: {total_valid}\n"
        f"Range [{min_depth}, 1): {count_low} pixels ({count_low / total_valid * 100:.2f}%)\n"
        f"Range [1, 10): {count_mid} pixels ({count_mid / total_valid * 100:.2f}%)\n"
        f"Range [10, {max_depth}]: {count_high} pixels ({count_high / total_valid * 100:.2f}%)\n"
    )
    print(stats_text)

    with open(os.path.join(output_dir, 'depth_stats.txt'), 'w') as f:
        f.write(stats_text)

    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(depth_valid, bins=100, range=(min_depth, max_depth),
                                density=True, alpha=0.6, color='skyblue', label='Histogram')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.plot(bin_centers, n, 'r-', linewidth=2, label='Density Curve')

    plt.title(f'Pixel Depth Distribution (Range: {min_depth}-{max_depth}m)')
    plt.xlabel('Depth (meters)')
    plt.ylabel('Density (Count Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, 'depth_distribution.png'))
    plt.close()
    print(f"Depth analysis plots saved to {output_dir}")


# ==================== [NEW] Save test/train poses to npy folder ====================
# Test poses: {model}_{scene}.npy        e.g. 'nope-nerf - adj - loss_shoulun.npy'
# Train poses: {model}_{scene}_train.npy  (for cross-model Sim(3) alignment in eval_NVS.py)
# model = root directory name of the current project (repo), independent of out_dir and cwd
# scene = config file name without extension, e.g. configs/LLFF/shoulun.yaml -> shoulun
DEFAULT_POSE_DIR = '/home/c1237/zjb/npy'


def get_repo_model_name():
    """e.g. project located at '/home/c1237/zjb/nope-nerf - adj - loss'
    -> returns 'nope-nerf - adj - loss'"""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.basename(repo_root)


def save_poses_for_cross_render(cfg, config_path, eval_c2ws,
                                learned_pose_param_net, train_dataset):
    if not (cfg['pose']['learn_pose'] and cfg['eval_pose']['type_to_eval'] != 'train'):
        return

    pose_dir = cfg['eval_pose'].get('pose_save_dir', DEFAULT_POSE_DIR)
    model_name = get_repo_model_name()
    scene_name = os.path.splitext(os.path.basename(config_path))[0]

    os.makedirs(pose_dir, exist_ok=True)

    test_path = os.path.join(pose_dir, f'{model_name}_{scene_name}.npy')
    np.save(test_path, eval_c2ws.detach().cpu().numpy())

    with torch.no_grad():
        learned_train = torch.stack(
            [learned_pose_param_net(i) for i in range(train_dataset['img'].N_imgs_train)])
    train_path = os.path.join(pose_dir, f'{model_name}_{scene_name}_train.npy')
    np.save(train_path, learned_train.detach().cpu().numpy())

    print(f'[pose-save] model="{model_name}" scene="{scene_name}"')
    print(f'[pose-save] test poses  -> {test_path}')
    print(f'[pose-save] train poses -> {train_path}')
# ==============================================================================


# ==================== [NEW] LLFF poses_bounds format [N,17] npy ====================
# Each row has 17 dimensions, in the following order:
#   0-2:  r11 r12 r13      3: tx       4: H
#   5-7:  r21 r22 r23      8: ty       9: W
#   10-12: r31 r32 r33    13: tz      14: f
#   15: near(=0)         16: far(=1)
# Merged file: test set at sequence indices 8x+4 (x=0,1,2,... -> 4,12,20,...),
# train set fills the remaining positions in order.
def encode_pose_bounds_row(c2w, H, W, f):
    """Encode the c2w (4x4), H, W, f of one image into a 17-dim row"""
    row = np.zeros(17, dtype=np.float32)
    row[0:3] = c2w[0, 0:3]      # r11 r12 r13
    row[3] = c2w[0, 3]          # tx
    row[4] = H                  # image height H
    row[5:8] = c2w[1, 0:3]      # r21 r22 r23
    row[8] = c2w[1, 3]          # ty
    row[9] = W                  # image width W
    row[10:13] = c2w[2, 0:3]    # r31 r32 r33
    row[13] = c2w[2, 3]         # tz
    row[14] = f                 # focal length (pixels)
    row[15] = 0.0               # near
    row[16] = 1.0               # far
    return row


def test_image_index(x):
    """Test-set positions in the merged image sequence: x=0,1,2,... -> 4, 12, 20, 28, ...
    (first test image at index 4, every 8th image)"""
    return 8 * x + 4


def save_poses_bounds_combined(cfg, config_path, eval_c2ws,
                               learned_pose_param_net, train_dataset,
                               eval_dataset, fxfy):
    """After training, encode train+test c2w/H/W/f into LLFF format [N,17] npy.
    Test set placed at 8x+4 positions, train set fills the rest. near=0, far=1.
    H/W/f are restored to the original resolution before resize_factor downsampling."""
    if not (cfg['pose']['learn_pose'] and cfg['eval_pose']['type_to_eval'] != 'train'):
        return

    pose_dir = cfg['eval_pose'].get('pose_save_dir', DEFAULT_POSE_DIR)
    scene_name = os.path.splitext(os.path.basename(config_path))[0]
    os.makedirs(pose_dir, exist_ok=True)

    # ---------- Downsampling factor: H/W/f must be scaled back to original resolution ----------
    try:
        resize_factor = cfg['dataloading']['resize_factor']
    except KeyError:
        resize_factor = 1
        print('[poses-bounds] Warning: dataloading.resize_factor not found in cfg, defaulting to 1')

    # ---------- Train poses (same convention as save_poses_for_cross_render) ----------
    n_train = train_dataset['img'].N_imgs_train
    H_tr = train_dataset['img'].H * resize_factor   # original image height
    W_tr = train_dataset['img'].W * resize_factor   # original image width
    with torch.no_grad():
        train_c2ws = torch.stack(
            [learned_pose_param_net(i) for i in range(n_train)]).detach().cpu().numpy()

    # ---------- Test poses (eval_c2ws already optimized by opt-pose) ----------
    test_c2ws = eval_c2ws.detach().cpu().numpy()
    n_test = test_c2ws.shape[0]
    H_te = eval_dataset['img'].H * resize_factor
    W_te = eval_dataset['img'].W * resize_factor

    # ---------- Focal length (convert via working resolution, then restore to original pixels) ----------
    if fxfy is not None:
        f_tr = float(fxfy[0].item()) * train_dataset['img'].W / 2.0 * resize_factor
        f_te = float(fxfy[0].item()) * eval_dataset['img'].W / 2.0 * resize_factor
    else:
        f_tr = float(train_dataset['img'].focal) * resize_factor
        f_te = float(eval_dataset['img'].focal) * resize_factor

    # ---------- Merge: test at 8x+4 positions, train fills the rest in order ----------
    n_all = n_train + n_test
    last_test_idx = test_image_index(n_test - 1)
    assert last_test_idx < n_all, (
        f'Test index {last_test_idx} exceeds merged sequence length {n_all}, '
        f'please check the number of train/test images (n_train={n_train}, n_test={n_test})')

    combined = np.zeros((n_all, 17), dtype=np.float32)
    i_tr, i_te = 0, 0
    for j in range(n_all):
        if i_te < n_test and j == test_image_index(i_te):
            combined[j] = encode_pose_bounds_row(test_c2ws[i_te], H_te, W_te, f_te)
            i_te += 1
        else:
            assert i_tr < n_train, 'Not enough train poses to fill by the 8x+4 rule'
            combined[j] = encode_pose_bounds_row(train_c2ws[i_tr], H_tr, W_tr, f_tr)
            i_tr += 1
    assert i_tr == n_train and i_te == n_test

    # ---------- Fixed file name poses_bounds.npy and print the absolute path ----------
    out_path = os.path.abspath(os.path.join(pose_dir, 'poses_bounds.npy'))
    np.save(out_path, combined)

    print(f'[poses-bounds] scene="{scene_name}", resize_factor={resize_factor}')
    print(f'[poses-bounds] n_train={n_train}, n_test={n_test}, total={n_all}')
    print(f'[poses-bounds] test indices (8x+4): '
          f'{[test_image_index(x) for x in range(min(n_test, 8))]}{" ..." if n_test > 8 else ""}')
    print(f'[poses-bounds] H={H_tr}, W={W_tr}, f={f_tr:.2f} (original resolution)')
    print(f'[poses-bounds] File saved: {out_path}')


def eval(cfg, config_path):
    torch.manual_seed(0)
    is_cuda = (torch.cuda.is_available())
    device = torch.device("cuda" if is_cuda else "cpu")

    out_dir = cfg['training']['out_dir']
    generation_dir = os.path.join(out_dir, cfg['eval_pose']['extraction_dir'])
    if not os.path.exists(generation_dir):
        os.makedirs(generation_dir)
    log_out_dir = os.path.join(out_dir, 'logs')
    writer = SummaryWriter(log_out_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    file_handler = logging.FileHandler(os.path.join(generation_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Model
    network_type = cfg['model']['network_type']
    if network_type == 'official':
        model = mdl.OfficialStaticNerf(cfg)

    rendering_cfg = cfg['rendering']
    renderer = mdl.Renderer(model, rendering_cfg, device=device)

    # init model
    nope_nerf = mdl.get_model(renderer, cfg, device=device)

    checkpoint_io = CheckpointIO(out_dir, model=nope_nerf)  # changed
    # Dataloading
    train_loader, train_dataset = get_dataloader(cfg, mode='train', shuffle=False)
    eval_loader, eval_dataset = get_dataloader(cfg, mode='eval', shuffle=False)
    checkpoint_io.load(cfg['extract_images']['model_file'])
    use_learnt_poses = cfg['pose']['learn_pose']
    use_learnt_focal = cfg['pose']['learn_focal']
    num_epoch = cfg['eval_pose']['opt_pose_epoch']
    init_method = cfg['eval_pose']['init_method']
    opt_eval_lr = cfg['eval_pose']['opt_eval_lr']

    if cfg['eval_pose']['type_to_eval'] == 'train':
        N_imgs = train_dataset['img'].N_imgs
        img_list = train_dataset['img'].img_list
        loader = train_loader
        render_dir = os.path.join(generation_dir, 'eval_trained')
    else:
        N_imgs = eval_dataset['img'].N_imgs
        img_list = eval_dataset['img'].img_list
        loader = eval_loader
        render_dir = os.path.join(generation_dir, 'eval', init_method)

    if use_learnt_focal:
        focal_net = mdl.LearnFocal(cfg['pose']['learn_focal'], cfg['pose']['fx_only'], order=cfg['pose']['focal_order'])
        checkpoint_io_focal = mdl.CheckpointIO(out_dir, model=focal_net)
        checkpoint_io_focal.load(cfg['extract_images']['model_file_focal'])
        fxfy = focal_net(0)
        print('learned fx: {0:.2f}, fy: {1:.2f}'.format(fxfy[0].item(), fxfy[1].item()))
    else:
        focal_net = None
        fxfy = None

    if use_learnt_poses:
        if cfg['pose']['init_pose']:
            init_pose = train_dataset['img'].c2ws  # init with colmap
        else:
            init_pose = None
        learned_pose_param_net = mdl.LearnPose(train_dataset['img'].N_imgs, cfg['pose']['learn_R'],
                                               cfg['pose']['learn_t'], cfg=cfg, init_c2w=init_pose).to(device=device)
        checkpoint_io_pose = mdl.CheckpointIO(out_dir, model=learned_pose_param_net)
        checkpoint_io_pose.load(cfg['extract_images']['model_file_pose'])
        if cfg['eval_pose']['type_to_eval'] == 'train':
            eval_pose_param_net = learned_pose_param_net
        else:
            with torch.no_grad():
                init_c2ws = eval_dataset['img'].c2ws.to(device)
                learned_c2ws_train = torch.stack(
                    [learned_pose_param_net(i) for i in range(train_dataset['img'].N_imgs_train)])
                colmap_c2ws_train = train_dataset['img'].c2ws  # (N, 4, 4)
                colmap_c2ws_train = colmap_c2ws_train.to(device)
                if init_method == 'scale':
                    init_c2ws, scale_colmap2est = align_scale_c2b_use_a2b(colmap_c2ws_train, learned_c2ws_train,
                                                                          init_c2ws.clone())
                elif init_method == 'ate':
                    init_c2ws = align_ate_c2b_use_a2b(colmap_c2ws_train, learned_c2ws_train, init_c2ws)
                elif init_method == 'pre':
                    sample_rate = train_dataset['img'].sample_rate
                    init_c2ws = learned_c2ws_train[int(sample_rate / 2) - 1::sample_rate - 1][:N_imgs]
                elif init_method == 'none':
                    init_c2ws = None
            eval_pose_param_net = mdl.LearnPose(eval_dataset['img'].N_imgs, learn_R=True, learn_t=True, cfg=cfg,
                                                init_c2w=init_c2ws).to(device=device)
        optimizer_eval_pose = torch.optim.Adam(eval_pose_param_net.parameters(), lr=opt_eval_lr)
        scheduler_eval_pose = torch.optim.lr_scheduler.MultiStepLR(optimizer_eval_pose,
                                                                   milestones=list(
                                                                       range(0, int(num_epoch), int(num_epoch / 5))),
                                                                   gamma=0.5)
        '''Optimise eval poses'''
        if cfg['eval_pose']['type_to_eval'] != 'train':
            eval_pose_cfg = cfg['eval_pose']
            trainer = mdl.Trainer_pose(nope_nerf, eval_pose_cfg, device=device, optimizer_pose=optimizer_eval_pose,
                                       pose_param_net=eval_pose_param_net, focal_net=focal_net)
            for epoch_i in tqdm(range(num_epoch), desc='optimising eval'):
                L2_loss_epoch = []
                psnr_epoch = []
                for batch in eval_loader:
                    losses = trainer.train_step(batch)
                    L2_loss_epoch.append(losses['loss'].item())
                L2_loss_mean = np.mean(L2_loss_epoch)
                opt_pose_psnr = mse2psnr(L2_loss_mean)
                scheduler_eval_pose.step()

                writer.add_scalar('opt/psnr', opt_pose_psnr, epoch_i)

                tqdm.write(
                    '{0:6d} ep: Opt: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, L2_loss_mean, opt_pose_psnr))
    eval_pose_param_net.eval()
    eval_c2ws = torch.stack([eval_pose_param_net(i) for i in range(N_imgs)])

    # ==================== [NEW] Save poses (test set + train set) ====================
    save_poses_for_cross_render(cfg, config_path, eval_c2ws,
                                learned_pose_param_net, train_dataset)
    # ======================================================================

    # ============ [NEW] Save merged LLFF-format [N,17] poses_bounds npy ============
    save_poses_bounds_combined(cfg, config_path, eval_c2ws,
                               learned_pose_param_net, train_dataset,
                               eval_dataset, fxfy)
    # ==========================================================================

    # Generator
    generator = Eval_Images(
        renderer, cfg, use_learnt_poses=use_learnt_poses,
        use_learnt_focal=use_learnt_focal,
        device=device, render_type=cfg['rendering']['type'], c2ws=eval_c2ws, img_list=img_list
    )

    if not os.path.exists(render_dir):
        os.makedirs(render_dir)

    imgs = []
    depths = []
    eval_mse_list = []
    eval_psnr_list = []
    eval_ssim_list = []
    eval_lpips_list = []
    depth_gts = []

    all_depth_preds_flat = []

    depth_preds = []
    lpips_metric = lpips_lib.LPIPS(net='vgg').to(device)

    min_depth = 0.1
    max_depth = 20

    for data in loader:
        out = generator.eval_images(data, render_dir, fxfy, lpips_metric, logger=logger, min_depth=min_depth,
                                    max_depth=max_depth)
        imgs.append(out['img'])
        depths.append(out['depth'])
        eval_mse_list.append(out['mse'])
        eval_psnr_list.append(out['psnr'])
        eval_ssim_list.append(out['ssim'])
        eval_lpips_list.append(out['lpips'])
        depth_preds.append(out['depth_pred'])

        if isinstance(out['depth_pred'], torch.Tensor):
            all_depth_preds_flat.append(out['depth_pred'].detach().cpu().flatten())
        else:
            all_depth_preds_flat.append(torch.from_numpy(out['depth_pred']).flatten())

        depth_gts.append(out['depth_gt'])

    all_depth_preds_tensor = torch.cat(all_depth_preds_flat, dim=0)
    save_depth_path = os.path.join(render_dir, 'all_pixels_depth.pt')
    torch.save(all_depth_preds_tensor, save_depth_path)
    print(f"\nSaved all pixel depths to: {save_depth_path}")

    analyze_and_plot_depth(all_depth_preds_tensor, render_dir, min_depth=min_depth, max_depth=max_depth)

    mean_mse = np.mean(eval_mse_list)
    mean_psnr = np.mean(eval_psnr_list)
    mean_ssim = np.mean(eval_ssim_list)
    mean_lpips = np.mean(eval_lpips_list)
    print('--------------------------')
    print('Mean MSE: {0:.3f}, PSNR: {1:.3f}, SSIM: {2:.3f}, LPIPS {3:.3f}'.format(mean_mse, mean_psnr,
                                                                                  mean_ssim, mean_lpips))

    print("{0:.2f}".format(mean_psnr), '&' "{0:.3f}".format(mean_ssim), '&', "{0:.3f}".format(mean_lpips))

    if cfg['extract_images']['eval_depth']:
        depth_errors = []
        ratio = np.median(np.concatenate(depth_gts)) / \
                np.median(np.concatenate(depth_preds))
        for i in range(len(depth_preds)):
            gt_depth = depth_gts[i]
            pred_depth = depth_preds[i]

            pred_depth *= ratio
            pred_depth[pred_depth < min_depth] = min_depth
            pred_depth[pred_depth > max_depth] = max_depth

            depth_errors.append(compute_errors(gt_depth, pred_depth))

        mean_errors = np.array(depth_errors).mean(0)
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")

        with open(os.path.join(generation_dir, 'depth_evaluation.txt'), 'a') as f:
            f.writelines(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + '\n')
            f.writelines(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    imgs = np.stack(imgs, axis=0)
    video_out_dir = os.path.join(render_dir, 'video_out')
    if not os.path.exists(video_out_dir):
        os.makedirs(video_out_dir)
    imageio.mimwrite(os.path.join(video_out_dir, 'img.mp4'), imgs, fps=30, quality=9)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract images.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/default.yaml')
    eval(cfg, args.config)