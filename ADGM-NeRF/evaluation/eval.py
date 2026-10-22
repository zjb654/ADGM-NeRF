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

# matplotlib is used for plotting
import matplotlib.pyplot as plt


def analyze_and_plot_depth(depth_preds_tensor, output_dir, min_depth=1e-6, max_depth=100.0):
    """
    Analyze depth data, generate histograms and curve plots, and print statistics.
    The min_depth and max_depth parameters limit the analysis range.
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


# ==================== [NEW] Save test/train poses to the npy folder ====================
# Test poses:  {model_name}_{scene_name}.npy       e.g. 'nope-nerf - adj - loss_shoulun.npy'
# Train poses: {model_name}_{scene_name}_train.npy (used by eval_NVS.py for cross-model Sim(3) alignment)
# model_name = root directory name of the current project (repo), independent of out_dir and the run cwd
# scene_name = config file name without extension, e.g. configs/LLFF/shoulun.yaml -> shoulun
DEFAULT_POSE_DIR = '/home/c1237/zjb/npy'


def get_repo_model_name():
    """e.g. if the project is located at '/home/c1237/zjb/nope-nerf - adj - loss',
    this returns 'nope-nerf - adj - loss'"""
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
                    init_c2ws = learned_c2ws_train[2::sample_rate-1][:N_imgs]
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

    # ==================== [NEW] Save poses (test set + training set) ====================
    save_poses_for_cross_render(cfg, config_path, eval_c2ws,
                                learned_pose_param_net, train_dataset)
    # ======================================================================

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