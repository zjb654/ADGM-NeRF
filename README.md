# ADGM-NeRF

Official implementation of

> Zhang et al., "ADGM-NeRF: Adaptive Inverse Depth Annealing and Global
> Matching for Pose-Free Neural Radiance Fields in Low-Texture Scenes,"
> *PLOS ONE* (Manuscript PONE-D-26-28831), 2026.

ADGM-NeRF performs joint optimization of camera poses and neural radiance
fields without any pose prior, targeting low-texture scenes where
feature-based SfM pipelines (e.g., COLMAP) degrade or fail. The framework
combines three components:

1. **Inverse depth parameterization (IDP):** learnable scale-shift alignment
   of monocular depth priors performed in inverse depth space, improving
   near-field geometric detail while stabilizing distant regions.
2. **Adaptive depth loss annealing (ADLA):** an inverse-sigmoid schedule that
   smoothly transitions from strong geometric prior guidance to
   network-driven adaptive learning.
3. **Explicit global geometric matching (EGGM):** cross-frame 3D geometric
   consistency constraints built from opacity-peak surface points, with
   periodically refreshed, gradient-detached historical anchors that suppress
   long-term pose drift.

## Repository Structure

This repository contains two codebases:

| Folder         | Role |
| -------------- | ---- |
| `ADGM-NeRF/`   | **Main code** of the proposed method (training, evaluation, configs). |
| `NoPe-NeRF++/` | **Independent reproduction of NoPe-NeRF++**, released to support the baseline comparison experiments of the paper. All NoPe-NeRF++ results reported in the paper were produced with this code. See `NoPe-NeRF++/README.md` for details and the reproduction disclaimer. |

The two codebases follow the **same evaluation protocol**, ensuring a
consistent and fair comparison.

## Installation

```bash
cd ADGM-NeRF
conda env create -f environment.yaml
conda activate nope-nerf
```

The same environment also works for the `NoPe-NeRF++/` reproduction.

## Data

  **LLFF**, **CO3D V2**, **Self-collected low-texture dataset** (flange, handwheel, ross)
  and the model in NoPe-NeRF++ used to produce the reported results are
  hosted on Figshare:

  **https://figshare.com/articles/dataset/ADGM-NeRF/32566935**

  Monocular depth priors are generated with DPT; see
  `ADGM-NeRF/configs/preprocess.yaml` and `ADGM-NeRF/preprocess/dpt_depth.py`.

## Training

Per-scene configuration files are provided under `ADGM-NeRF/configs/`
(`LLFF/`, `co3d/`, `Tanks/`, etc.). Train a scene with:

```bash
cd ADGM-NeRF
python train.py --config configs/LLFF/fern.yaml
```

Global matching points (EGGM anchors) are refreshed every 50,000 iterations
by default; the viewing-angle constraint, distance ratio, anchor capacity and
update interval can be adjusted in the config files.

## Evaluation

Camera-pose metrics (RPE / ATE) and novel-view-synthesis metrics
(PSNR / SSIM / LPIPS) are computed by `ADGM-NeRF/evaluation/eval.py` and
`ADGM-NeRF/evaluation/eval_poses.py`.

The evaluation protocol follows NoPe-NeRF: test frames are excluded from
training, and each test pose is initialized from its preceding training frame
and refined by minimizing photometric error on the test image.

## NoPe-NeRF++ Baseline (Independent Reproduction)

The `NoPe-NeRF++/` folder contains our independent re-implementation of

> Shi et al., "NoPe-NeRF++: Local-to-Global Optimization of NeRF with No Pose
> Prior," *Computer Graphics Forum*, 2025, 44(2): e70012.

**This is NOT the official code released by the original authors.** At the
time of our experiments, no official implementation of NoPe-NeRF++ was
publicly available to us. We re-implemented its core framework from the
algorithmic descriptions in the original paper, including local-to-global
window optimization, multi-view point tracking, and geometric consistency
losses. All results produced with this code should be regarded as a
**third-party reproduction** and may differ from the numbers reported in the
original NoPe-NeRF++ paper. Any discrepancy should be attributed to this
reproduction, not to the original authors.

Usage is identical to the main code; see `NoPe-NeRF++/README.md`.

## Citation

If you use this code or the reported results, please cite:

```bibtex
@article{zhang2026adgmnerf,
  title   = {{ADGM-NeRF}: Adaptive Inverse Depth Annealing and Global Matching
             for Pose-Free Neural Radiance Fields in Low-Texture Scenes},
  author  = {Zhang, Jian-Bin and Ding, Jin and Xu, Ya-Bo and Wan, Li-Wei
             and Sun, Yong-Zhi and Tan, Ping and Ma, Ji-En and Fang, You-Tong},
  journal = {PLOS ONE},
  note    = {Manuscript PONE-D-26-28831},
  year    = {2026}
}

@article{shi2025nopenerfpp,
  title   = {{NoPe-NeRF++}: Local-to-Global Optimization of NeRF with No Pose Prior},
  author  = {Shi, Dian and Cao, Shuo and Wu, Bangyan and others},
  journal = {Computer Graphics Forum},
  volume  = {44},
  number  = {2},
  pages   = {e70012},
  year    = {2025}
}
```

## Acknowledgement

Both codebases build on the public
[NoPe-NeRF](https://github.com/ActiveVisionLab/nope-nerf) implementation
(Bian et al., CVPR 2023). We thank the authors for releasing their code.
