"""Evaluation script for GalaxEye binary change detection."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import rasterio
import yaml

from src.dataset import ChangeDetectionDataset, get_val_transform
from src.model   import SiameseUNet
from src.metrics import ConfusionMeter
from src.utils   import get_logger, set_seed, get_device, load_checkpoint
from torch.utils.data import DataLoader


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="GalaxEye Change Detection — Evaluation")
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to split directory (e.g. Dataset/test)")
    p.add_argument("--weights",   type=str, required=True,
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--config",    type=str, default="config.yaml")
    p.add_argument("--threshold", type=float, default=None,
                   help="Sigmoid threshold (default from config)")
    p.add_argument("--n_vis",     type=int, default=10,
                   help="Number of qualitative samples to save")
    return p.parse_args()


# Confusion matrix

def plot_confusion_matrix(meter: ConfusionMeter, split: str, out_path: Path):
    cm = meter.confusion_matrix()   # [[TN, FP], [FN, TP]]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["Pred No-Change", "Pred Change"],
        yticklabels=["GT No-Change",   "GT Change"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {split.upper()}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# Qualitative samples

def save_qualitative(
    dataset: ChangeDetectionDataset,
    model: torch.nn.Module,
    device: torch.device,
    out_dir: Path,
    n: int = 10,
    threshold: float = 0.5,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = list(range(min(n, len(dataset))))

    model.eval()
    cmap_mask = matplotlib.colors.ListedColormap(["#1a1a2e", "#e94560"])

    for i, idx in enumerate(indices):
        pre_t, post_t, mask_t = dataset[idx]

        with torch.no_grad():
            logit = model(
                pre_t.unsqueeze(0).to(device),
                post_t.unsqueeze(0).to(device),
            )
            prob = torch.sigmoid(logit).squeeze().cpu().numpy()
            pred = (prob >= threshold).astype(np.uint8)

        mask_np = mask_t.numpy()

        # Convert tensors to display-ready numpy
        pre_np  = pre_t.permute(1, 2, 0).numpy()    # (H,W,3)
        post_np = post_t.squeeze(0).numpy()           # (H,W)

        # Compute IoU for this sample
        tp = ((pred == 1) & (mask_np == 1)).sum()
        fp = ((pred == 1) & (mask_np == 0)).sum()
        fn = ((pred == 0) & (mask_np == 1)).sum()
        sample_iou = tp / (tp + fp + fn + 1e-8)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(
            f"Sample {idx} — IoU={sample_iou:.3f}  "
            f"({'Success' if sample_iou > 0.3 else 'Failure'})",
            fontsize=11, fontweight="bold",
        )

        axes[0].imshow(np.clip(pre_np, 0, 1))
        axes[0].set_title("Pre-event (EO)", fontsize=9); axes[0].axis("off")

        axes[1].imshow(post_np, cmap="gray")
        axes[1].set_title("Post-event (SAR)", fontsize=9); axes[1].axis("off")

        axes[2].imshow(mask_np, cmap=cmap_mask, vmin=0, vmax=1)
        axes[2].set_title("Ground Truth", fontsize=9); axes[2].axis("off")

        axes[3].imshow(pred, cmap=cmap_mask, vmin=0, vmax=1)
        axes[3].set_title("Prediction", fontsize=9); axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / f"sample_{i:03d}.png", dpi=120, bbox_inches="tight")
        plt.close()


# Main evaluation

def main():
    args   = parse_args()
    logger = get_logger("eval")

    # Config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device    = get_device()
    threshold = args.threshold or cfg["eval"]["threshold"]
    img_size  = tuple(cfg["dataset"]["image_size"])

    logger.info(f"Device    : {device}")
    logger.info(f"Data path : {args.data_path}")
    logger.info(f"Weights   : {args.weights}")
    logger.info(f"Threshold : {threshold}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = ChangeDetectionDataset(
        args.data_path,
        transform=get_val_transform(),
        img_size=img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
    )
    logger.info(f"Samples: {len(dataset)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = SiameseUNet(pretrained=False).to(device)
    load_checkpoint(args.weights, model, device=device)
    model.eval()
    logger.info("Model loaded.")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    meter = ConfusionMeter(threshold=threshold)

    with torch.no_grad():
        for pre, post, mask in loader:
            pre  = pre.to(device)
            post = post.to(device)
            mask = mask.to(device)
            logits = model(pre, post)
            meter.update(logits, mask)

    metrics = meter.compute()
    cm      = meter.confusion_matrix()

    logger.info("\n" + "=" * 50)
    logger.info(f"  IoU       : {metrics['iou']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  F1        : {metrics['f1']:.4f}")
    logger.info(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    logger.info("=" * 50)

    # ── Save results ─────────────────────────────────────────────────────────
    split      = Path(args.data_path).name
    results_dir = Path(cfg["output"]["results_dir"]); results_dir.mkdir(exist_ok=True)

    out_json = results_dir / f"metrics_{split}.json"
    with open(out_json, "w") as f:
        json.dump({"split": split, "threshold": threshold, **metrics}, f, indent=2)
    logger.info(f"Metrics saved → {out_json}")

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    plot_confusion_matrix(meter, split, results_dir / f"confusion_matrix_{split}.png")

    # ── Qualitative visualisations ───────────────────────────────────────────
    n_vis   = args.n_vis
    vis_dir = results_dir / f"qualitative_{split}"
    logger.info(f"Saving {n_vis} qualitative samples → {vis_dir}")
    save_qualitative(dataset, model, device, vis_dir, n=n_vis, threshold=threshold)

    logger.info("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
