"""
explore_data.py
───────────────
Exploratory Data Analysis for GalaxEye EO-SAR dataset.
Checks channel distribution, resolution, and class imbalance.
"""

import rasterio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_split(split_path: str):
    root = Path(split_path)
    pre_dir = root / "pre-event"
    post_dir = root / "post-event"
    tgt_dir = root / "target"
    
    files = sorted(list(pre_dir.glob("*.tif")))
    print(f"Analyzing {split_path}: {len(files)} samples")
    
    pixel_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for f in files[:100]:  # sample 100 for speed
        with rasterio.open(tgt_dir / f.name) as src:
            mask = src.read(1)
            unique, counts = np.unique(mask, return_counts=True)
            for u, c in zip(unique, counts):
                pixel_counts[u] += c
                
    total = sum(pixel_counts.values())
    print("\nClass Distribution (Raw):")
    for k, v in pixel_counts.items():
        print(f"  Class {k}: {v/total*100:.2f}%")
        
    # Remapped distribution
    no_change = pixel_counts[0] + pixel_counts[1]
    change = pixel_counts[2] + pixel_counts[3]
    print(f"\nRemapped Binary Distribution:")
    print(f"  No-Change (0): {no_change/total*100:.2f}%")
    print(f"  Change (1):    {change/total*100:.2f}%")
    print(f"  Imbalance Ratio: 1:{no_change/change:.1f}")

if __name__ == "__main__":
    import os
    if os.path.exists("Dataset/train"):
        analyze_split("Dataset/train")
    else:
        print("Dataset/train not found. Skipping EDA.")
