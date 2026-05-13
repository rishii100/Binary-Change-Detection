"""Loss functions for change detection."""


import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation (works on logits).
    smooth avoids division by zero.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) raw logits
            targets : (B, H, W)    binary long {0, 1}
        """
        probs   = torch.sigmoid(logits).squeeze(1)          # (B, H, W)
        targets = targets.float()

        intersection = (probs * targets).sum(dim=(1, 2))
        cardinality  = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=15.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        pw = torch.tensor([pos_weight], dtype=torch.float32)
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pw)
        
    def forward(self, logits, targets):
        bce_loss = self.bce(logits.squeeze(1), targets.float())
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """
    alpha * Focal(pos_weight) + beta * Dice

    Args:
        pos_weight : scalar weight for positive (Change) class in Focal Loss.
        alpha      : weight on Focal term
        beta       : weight on Dice term
        smooth     : Dice smoothing factor
    """

    def __init__(
        self,
        pos_weight: float = 15.0,
        alpha: float = 0.5,
        beta: float  = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha    = alpha
        self.beta     = beta
        self.dice     = DiceLoss(smooth=smooth)
        self.focal    = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W)
            targets : (B, H, W) long
        """
        # BCE needs float targets and squeezed logits or matching shape
        targets_f = targets.float()
        focal_loss  = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * focal_loss + self.beta * dice_loss


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, H, W   = 2, 256, 256
    logits    = torch.randn(B, 1, H, W)
    targets   = torch.randint(0, 2, (B, H, W))
    loss_fn   = CombinedLoss(pos_weight=58.4)
    loss      = loss_fn(logits, targets)
    print(f"Combined loss: {loss.item():.4f}  ✅")
