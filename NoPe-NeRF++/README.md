# NoPe-NeRF++ (Independent Reproduction)

This repository releases our **independent reproduction of NoPe-NeRF++**, which
was developed to support the baseline comparison experiments of our paper:

> Zhang et al., "ADGM-NeRF: Adaptive Inverse Depth Annealing and Global
> Matching for Pose-Free Neural Radiance Fields in Low-Texture Scenes,"
> *PLOS ONE* (Manuscript PONE-D-26-28831), 2026.

All NoPe-NeRF++ results reported in that paper were produced with this code.

## Important Notice

This is **NOT** the official code released by the original authors of
NoPe-NeRF++:

> Shi et al., "NoPe-NeRF++: Local-to-Global Optimization of NeRF with No Pose
> Prior," *Computer Graphics Forum*, 2025, 44(2): e70012.

At the time of our experiments, no official implementation of NoPe-NeRF++ was
publicly available to us. We therefore re-implemented its core framework from
the algorithmic descriptions in the original paper, including local-to-global
window optimization, multi-view point tracking, and geometric consistency
losses. The codebase builds on the public
[NoPe-NeRF](https://github.com/ActiveVisionLab/nope-nerf) implementation
(Bian et al., CVPR 2023).

All results produced with this code should be regarded as a **third-party
reproduction** and may differ from the numbers reported in the original
NoPe-NeRF++ paper. Any discrepancy should be attributed to this reproduction,
not to the original authors.

## Pre-trained Models

Pre-trained model checkpoints used to produce the reported results are hosted
on Figshare:

**https://figshare.com/articles/dataset/ADGM-NeRF/32566935**

Download the checkpoints and copy them into the corresponding output
directories of this repository (preserving the original directory structure),
then run evaluation with the corresponding config files under `configs/`.

## Installation

```bash
conda env create -f environment.yaml
conda activate <env_name>
```

## Training

```bash
python train.py --config configs/LLFF/fern.yaml
```

## Evaluation

Camera-pose metrics (RPE / ATE) and novel-view-synthesis metrics
(PSNR / SSIM / LPIPS) are computed by `evaluation/eval.py`,
`evaluation/eval_poses.py`, and `evaluation/eval_NVS.py`.
The evaluation protocol follows NoPe-NeRF: test frames are excluded from
training, and each test pose is initialized from its preceding training frame
and refined by minimizing photometric error on the test image.

## Citation

If you use this code or the reported reproduction results, please cite both
the original NoPe-NeRF++ paper and our work:

```bibtex
@article{shi2025nopenerfpp,
  title   = {{NoPe-NeRF++}: Local-to-Global Optimization of NeRF with No Pose Prior},
  author  = {Shi, Dian and Cao, Shuo and Wu, Bangyan and others},
  journal = {Computer Graphics Forum},
  volume  = {44},
  number  = {2},
  pages   = {e70012},
  year    = {2025}
}

@article{zhang2026adgmnerf,
  title   = {{ADGM-NeRF}: Adaptive Inverse Depth Annealing and Global Matching
             for Pose-Free Neural Radiance Fields in Low-Texture Scenes},
  author  = {Zhang, Jian-Bin and Ding, Jin and Xu, Ya-Bo and Wan, Li-Wei
             and Sun, Yong-Zhi and Tan, Ping and Ma, Ji-En and Fang, You-Tong},
  journal = {PLOS ONE},
  note    = {Manuscript PONE-D-26-28831},
  year    = {2026}
}
```
