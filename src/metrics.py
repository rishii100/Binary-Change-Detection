"""Metrics for change detection."""


import numpy as np
import torch


# Metrics functions

def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Args:
        preds   : (B, 1, H, W) logits  OR  (B, H, W) probabilities
        targets : (B, H, W) long binary {0, 1}
        threshold : sigmoid threshold

    Returns dict with keys: iou, precision, recall, f1, tp, fp, fn, tn
    """
    if preds.dim() == 4:
        probs = torch.sigmoid(preds).squeeze(1)
    else:
        probs = preds

    pred_bin = (probs >= threshold).long()
    tgt      = targets.long()

    tp = ((pred_bin == 1) & (tgt == 1)).sum().item()
    fp = ((pred_bin == 1) & (tgt == 0)).sum().item()
    fn = ((pred_bin == 0) & (tgt == 1)).sum().item()
    tn = ((pred_bin == 0) & (tgt == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)

    return {
        "iou"      : iou,
        "precision": precision,
        "recall"   : recall,
        "f1"       : f1,
        "tp"       : tp,
        "fp"       : fp,
        "fn"       : fn,
        "tn"       : tn,
    }


# Accumulating meter

class ConfusionMeter:
    """Accumulates TP/FP/FN/TN over batches, then computes final metrics."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        m = compute_metrics(logits, targets, self.threshold)
        self.tp += m["tp"]
        self.fp += m["fp"]
        self.fn += m["fn"]
        self.tn += m["tn"]

    def compute(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        iou       = tp / (tp + fp + fn + 1e-8)
        return {
            "iou"      : round(iou, 4),
            "precision": round(precision, 4),
            "recall"   : round(recall, 4),
            "f1"       : round(f1, 4),
            "tp"       : tp,
            "fp"       : fp,
            "fn"       : fn,
            "tn"       : tn,
        }

    def confusion_matrix(self) -> np.ndarray:
        """Returns 2×2 confusion matrix [[TN, FP], [FN, TP]]."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preds   = torch.randn(2, 1, 256, 256)
    targets = torch.randint(0, 2, (2, 256, 256))
    m       = ConfusionMeter()
    m.update(preds, targets)
    print(m.compute())
    print("Confusion matrix:\n", m.confusion_matrix())
    print("✅ Metrics OK")
