"""Dataset for GalaxEye EO-SAR change detection."""


import random
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Label Remapping

LABEL_REMAP = {0: 0, 1: 0, 2: 1, 3: 1}

def remap_mask(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.uint8)
    for orig, remapped in LABEL_REMAP.items():
        out[arr == orig] = remapped
    return out


# Dataset class

class ChangeDetectionDataset(Dataset):
    """
    Loads (pre-event EO, post-event SAR, binary mask) triplets.

    Args:
        root      : path to split directory, e.g. Dataset/train
        transform : albumentations transform applied to (pre, post, mask) together
        img_size  : (H, W) to resize all images to; None = use native size
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        img_size: Optional[tuple[int, int]] = (256, 256),
    ):
        self.root      = Path(root)
        self.transform = transform
        self.img_size  = img_size

        pre_dir    = self.root / "pre-event"
        post_dir   = self.root / "post-event"
        target_dir = self.root / "target"

        pre_files = sorted(pre_dir.glob("*.tif"))
        self.triplets: list[tuple[Path, Path, Path]] = []
        for pre in pre_files:
            post   = post_dir   / pre.name
            target = target_dir / pre.name
            if post.exists() and target.exists():
                self.triplets.append((pre, post, target))

        assert len(self.triplets) > 0, f"No triplets found under {self.root}"

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        pre_p, post_p, tgt_p = self.triplets[idx]

        # ── Read TIFs ────────────────────────────────────────────────────────
        with rasterio.open(pre_p) as src:
            pre = src.read().astype(np.float32)       # (3, H, W)  uint8 → float

        with rasterio.open(post_p) as src:
            post = src.read().astype(np.float32)      # (1, H, W)

        with rasterio.open(tgt_p) as src:
            mask = src.read(1).astype(np.uint8)       # (H, W)

        # ── Label remap ──────────────────────────────────────────────────────
        mask = remap_mask(mask)                        # {0, 1}

        # ── Normalize to [0, 1] ──────────────────────────────────────────────
        pre  = pre  / 255.0
        post = post / 255.0

        # ── Reshape to (H, W, C) for albumentations ──────────────────────────
        pre_hw  = pre.transpose(1, 2, 0)              # (H, W, 3)
        post_hw = post.transpose(1, 2, 0)             # (H, W, 1)

        # ── Resize first (before augment) ────────────────────────────────────
        if self.img_size is not None:
            h, w = self.img_size
            resize = A.Resize(h, w, interpolation=1)
            result = resize(image=pre_hw, masks=[post_hw[:, :, 0], mask])
            pre_hw      = result["image"]                    # (H', W', 3)
            post_hw_ch  = result["masks"][0]                 # (H', W')
            mask        = result["masks"][1]                 # (H', W')
            post_hw     = post_hw_ch[:, :, None]             # (H', W', 1)

        # ── Albumentations transform ──────────────────────────────────────────
        if self.transform is not None:
            # stack pre (3ch) + post (1ch) as 4-ch image so spatial augs apply together
            combined = np.concatenate([pre_hw, post_hw], axis=-1)  # (H, W, 4)
            result   = self.transform(image=combined, mask=mask)
            combined = result["image"]                              # (H, W, 4)
            mask     = result["mask"]                               # (H, W)
            pre_hw   = combined[:, :, :3]
            post_hw  = combined[:, :, 3:4]

        # ── To tensors ───────────────────────────────────────────────────────
        pre_t  = torch.from_numpy(pre_hw.transpose(2, 0, 1))    # (3, H, W)
        post_t = torch.from_numpy(post_hw.transpose(2, 0, 1))   # (1, H, W)
        mask_t = torch.from_numpy(mask.astype(np.int64))        # (H, W) long

        return pre_t, post_t, mask_t

    def __repr__(self) -> str:
        return (f"ChangeDetectionDataset(root={self.root}, "
                f"n_samples={len(self)}, img_size={self.img_size})")


# Transforms

def get_train_transform() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.GaussNoise(p=0.3),
        A.RandomBrightnessContrast(p=0.4),
    ])


def get_val_transform() -> None:
    return None   # only resize, no augmentation (done inside Dataset)


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    ds = ChangeDetectionDataset("Dataset/train", transform=get_train_transform())
    pre, post, mask = ds[0]
    print(f"pre  : {pre.shape}  {pre.dtype}  min={pre.min():.3f}  max={pre.max():.3f}")
    print(f"post : {post.shape} {post.dtype}  min={post.min():.3f}  max={post.max():.3f}")
    print(f"mask : {mask.shape} {mask.dtype}  unique={mask.unique().tolist()}")
    print("✅ Dataset OK")
