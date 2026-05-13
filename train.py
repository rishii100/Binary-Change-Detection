"""Training script for GalaxEye binary change detection."""


import argparse
import json
import time
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset  import ChangeDetectionDataset, get_train_transform, get_val_transform
from src.model    import SiameseUNet, count_params
from src.losses   import CombinedLoss
from src.metrics  import ConfusionMeter
from src.utils    import get_logger, set_seed, get_device, save_checkpoint, AverageMeter


# CLI args

def parse_args():
    p = argparse.ArgumentParser(description="GalaxEye Change Detection — Training")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


# Main training loop

def main():
    args   = parse_args()
    logger = get_logger("train")

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {args.config}")

    # ── Seed + device ────────────────────────────────────────────────────────
    set_seed(cfg["seed"])
    device = get_device()
    logger.info(f"Device: {device}")

    # ── Directories ──────────────────────────────────────────────────────────
    ckpt_dir    = Path(cfg["output"]["checkpoint_dir"]); ckpt_dir.mkdir(exist_ok=True)
    results_dir = Path(cfg["output"]["results_dir"]);    results_dir.mkdir(exist_ok=True)
    log_dir     = Path(cfg["output"]["log_dir"]);        log_dir.mkdir(exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    img_size = tuple(cfg["dataset"]["image_size"])
    data_root = Path(cfg["dataset"]["root"])

    train_ds = ChangeDetectionDataset(
        data_root / "train",
        transform=get_train_transform(),
        img_size=img_size,
    )
    val_ds = ChangeDetectionDataset(
        data_root / "val",
        transform=get_val_transform(),
        img_size=img_size,
    )
    logger.info(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    pin_memory  = cfg["training"]["pin_memory"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = SiameseUNet(pretrained=cfg["model"]["pretrained"]).to(device)
    logger.info(f"Model params — {count_params(model)}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_cfg = cfg["loss"]
    criterion = CombinedLoss(
        pos_weight=loss_cfg["bce_pos_weight"],
        alpha=loss_cfg["alpha"],
        beta=loss_cfg["beta"],
        smooth=loss_cfg["smooth"],
    ).to(device)

    # ── Optimizer ────────────────────────────────────────────────────────────
    opt_cfg   = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
    )

    # ── Scheduler ────────────────────────────────────────────────────────────
    sched_cfg = cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sched_cfg["T_max"]
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_iou  = 0.0
    patience_cnt  = 0

    if args.resume:
        from src.utils import load_checkpoint
        ckpt = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch  = ckpt["epoch"] + 1
        best_val_iou = ckpt["metrics"].get("val_iou", 0.0)
        logger.info(f"Resumed from epoch {ckpt['epoch']} | best_val_iou={best_val_iou:.4f}")

    # ── TensorBoard ──────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(log_dir))

    # ── Training Loop ─────────────────────────────────────────────────────────
    n_epochs   = cfg["training"]["epochs"]
    clip_norm  = cfg["training"]["gradient_clip_norm"]
    patience   = cfg["training"]["early_stopping_patience"]
    threshold  = cfg["eval"]["threshold"]

    history = []

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        loss_meter = AverageMeter()
        train_meter = ConfusionMeter(threshold=threshold)

        for batch_idx, (pre, post, mask) in enumerate(train_loader):
            pre  = pre.to(device)
            post = post.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            logits = model(pre, post)            # (B, 1, H, W)
            loss   = criterion(logits, mask)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            loss_meter.update(loss.item(), n=pre.size(0))
            with torch.no_grad():
                train_meter.update(logits.detach(), mask)

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{n_epochs}  "
                    f"Step {batch_idx+1}/{len(train_loader)}  "
                    f"Loss={loss_meter.avg:.4f}"
                )

        train_metrics = train_meter.compute()
        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_meter = ConfusionMeter(threshold=threshold)
        val_loss_meter = AverageMeter()

        with torch.no_grad():
            for pre, post, mask in val_loader:
                pre  = pre.to(device)
                post = post.to(device)
                mask = mask.to(device)
                logits = model(pre, post)
                loss   = criterion(logits, mask)
                val_loss_meter.update(loss.item(), n=pre.size(0))
                val_meter.update(logits, mask)

        val_metrics = val_meter.compute()
        epoch_time  = time.time() - t0

        logger.info(
            f"Epoch {epoch+1}/{n_epochs}  ({epoch_time:.1f}s)  "
            f"TrainLoss={loss_meter.avg:.4f}  ValLoss={val_loss_meter.avg:.4f}  "
            f"Val IoU={val_metrics['iou']:.4f}  Val F1={val_metrics['f1']:.4f}  "
            f"Val Prec={val_metrics['precision']:.4f}  Val Rec={val_metrics['recall']:.4f}"
        )

        # ── TensorBoard ──────────────────────────────────────────────────────
        writer.add_scalar("Loss/train",      loss_meter.avg,          epoch)
        writer.add_scalar("Loss/val",        val_loss_meter.avg,      epoch)
        writer.add_scalar("IoU/train",       train_metrics["iou"],    epoch)
        writer.add_scalar("IoU/val",         val_metrics["iou"],      epoch)
        writer.add_scalar("F1/val",          val_metrics["f1"],       epoch)
        writer.add_scalar("Precision/val",   val_metrics["precision"], epoch)
        writer.add_scalar("Recall/val",      val_metrics["recall"],   epoch)
        writer.add_scalar("LR",              scheduler.get_last_lr()[0], epoch)

        # ── History ──────────────────────────────────────────────────────────
        row = {
            "epoch"      : epoch + 1,
            "train_loss" : round(loss_meter.avg, 6),
            "val_loss"   : round(val_loss_meter.avg, 6),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
            "epoch_time_s": round(epoch_time, 2),
        }
        history.append(row)

        # ── Checkpoint (every epoch) ─────────────────────────────────────────
        save_checkpoint(
            model, optimizer, epoch,
            {"val_iou": val_metrics["iou"], "val_f1": val_metrics["f1"]},
            ckpt_dir / "last.pth",
        )

        # ── Best checkpoint ───────────────────────────────────────────────────
        val_iou = val_metrics["iou"]
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_cnt = 0
            save_checkpoint(
                model, optimizer, epoch,
                {"val_iou": val_iou, "val_f1": val_metrics["f1"]},
                ckpt_dir / "best.pth",
            )
            logger.info(f"  ✅ New best val IoU: {val_iou:.4f} → checkpoints/best.pth")
        else:
            patience_cnt += 1
            logger.info(f"  No improvement. Patience {patience_cnt}/{patience}")
            if patience_cnt >= patience:
                logger.info("Early stopping triggered.")
                break

    # ── Save history ─────────────────────────────────────────────────────────
    hist_path = results_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    writer.close()
    logger.info(f"\n✅ Training complete. Best val IoU = {best_val_iou:.4f}")
    logger.info(f"   Checkpoint  : {ckpt_dir}/best.pth")
    logger.info(f"   History     : {hist_path}")


if __name__ == "__main__":
    main()
