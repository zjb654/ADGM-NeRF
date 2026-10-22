import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import gc
import yaml
import glob
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from DPT.dpt.models import DPTDepthModel
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision import transforms
from PIL import Image


def dpt_depth_simple(model_path, img_dir, save_dir, device):
    """Minimal memory-safe version: read images one by one without preloading."""

    # 1. Scan image files (without loading them)
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                       glob.glob(os.path.join(img_dir, '*.png')) +
                       glob.glob(os.path.join(img_dir, '*.JPG')))

    if len(img_files) == 0:
        raise ValueError(f"No images found in {img_dir}")

    print(f"Found {len(img_files)} images")

    # 2. Load the model
    print("Loading DPT model...")
    model = DPTDepthModel(
        path=model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    model.to(device)
    model.eval()

    # 3. Clear the cache after model loading
    torch.cuda.empty_cache()
    gc.collect()
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    # 4. Preprocessing pipeline
    net_w, net_h = 384, 384
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        Resize(net_w, net_h, resize_target=None, keep_aspect_ratio=True,
               ensure_multiple_of=32, resize_method="minimal"),
        normalization,
        PrepareForNet(),
    ])

    os.makedirs(save_dir, exist_ok=True)

    # 5. Process images one by one
    processed = 0
    for img_path in tqdm(img_files, desc="Generating depth"):
        try:
            # Read a single image
            img = np.array(Image.open(img_path).convert('RGB')) / 255.0

            # Preprocess
            sample = {'image': img}
            sample = transform(sample)
            img_tensor = torch.from_numpy(sample['image']).unsqueeze(0).to(device)

            # Inference (with AMP)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    depth = model(img_tensor)

            # Save the result
            depth = depth.squeeze().cpu().numpy()
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(save_dir, f'depth_{base_name}.npz')
            np.savez_compressed(save_path, pred=depth.astype(np.float32))

            processed += 1

            # Release memory immediately
            del img, img_tensor, depth, sample
            if processed % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue

    print(f"\n✓ Completed: {processed}/{len(img_files)} images")
    print(f"✓ Saved to: {save_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/preprocess.yaml'

    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # Resolve paths
    scene_name = cfg['dataloading']['scene'][0]
    base_path = cfg['dataloading']['path']

    # Image directory (following the NoPe-NeRF structure)
    img_dir = os.path.join(base_path, scene_name, 'images')
    if not os.path.exists(img_dir):
        # Try other common naming conventions
        img_dir = os.path.join(base_path, scene_name, 'imgs')

    save_dir = os.path.join(base_path, scene_name, 'dpt')
    model_path = cfg['depth']['path']

    print(f"Scene: {scene_name}")
    print(f"Images: {img_dir}")
    print(f"Output: {save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dpt_depth_simple(model_path, img_dir, save_dir, device)