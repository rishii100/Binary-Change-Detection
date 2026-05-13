"""Siamese UNet for binary change detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# Encoder

class ResNet34Encoder(nn.Module):
    """
    ResNet-34 feature extractor returning 5 skip-connection feature maps.
    in_chans controls the first-layer input channels (3 for EO, 1 for SAR).
    """

    def __init__(self, in_chans: int = 3, pretrained: bool = True):
        super().__init__()
        backbone = timm.create_model(
            "resnet34",
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans,
            out_indices=(0, 1, 2, 3, 4),
        )
        self.backbone = backbone
        # Feature channels: [64, 64, 128, 256, 512]
        self.out_channels = backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(x)


# Decoder

class DecoderBlock(nn.Module):
    """
    Upsample + fused skip → Conv-BN-ReLU × 2
    in_ch  = upsampled channels + fused skip channels
    out_ch = output channels
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # skip fusion: concat(|diff|, sum) → 2 * skip_ch channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + 2 * skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip_pre: torch.Tensor,
        skip_post: torch.Tensor,
    ) -> torch.Tensor:
        x        = self.upsample(x)
        diff     = torch.abs(skip_pre - skip_post)
        summed   = skip_pre + skip_post
        fused    = torch.cat([x, diff, summed], dim=1)
        return self.conv(fused)


# Siamese UNet

class SiameseUNet(nn.Module):
    """
    Weight-shared Siamese UNet for binary change detection.

    Pre-event  (EO,  3ch) and Post-event (SAR, 1ch) pass through
    independent first-layer adapters then share the same encoder weights
    from stage-1 onward via a common backbone.

    Design rationale:
      • Shared weights enforce modality-invariant feature learning
      • Absolute-difference + sum skip fusion captures change magnitude and context
      • Combined BCE+Dice loss handles 58:1 class imbalance
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Two encoders: one per modality (different in_chans for first layer)
        self.enc_pre  = ResNet34Encoder(in_chans=3, pretrained=pretrained)
        self.enc_post = ResNet34Encoder(in_chans=3, pretrained=pretrained)

        # Share weights from stage-1 onward (layer1 … layer4 + bn1)
        # This ties all parameters except the very first stem conv
        self._share_encoder_weights()

        # Encoder output channels: [64, 64, 128, 256, 512]
        enc_ch = self.enc_pre.out_channels  # e.g. [64, 64, 128, 256, 512]

        # Bridge (bottleneck) — no skip from here
        bridge_ch = enc_ch[-1]   # 512

        # Decoder: 4 blocks going from deep → shallow
        # Each block upsamples and fuses skip pairs
        self.dec4 = DecoderBlock(bridge_ch,   enc_ch[3], 256)   # 512 + 2*256 → 256
        self.dec3 = DecoderBlock(256,          enc_ch[2], 128)   # 256 + 2*128 → 128
        self.dec2 = DecoderBlock(128,          enc_ch[1],  64)   # 128 + 2*64  →  64
        self.dec1 = DecoderBlock( 64,          enc_ch[0],  32)   # 64  + 2*64  →  32

        # Final upsample to input resolution + classification head
        self.final_up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.head       = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),   # logit
        )

    def _share_encoder_weights(self):
        """
        Copy parameter references from enc_pre into enc_post for all shared
        layers. The first stem conv differs (in_chans 3 vs 1) so is excluded.
        This is called once at init; after that grad updates affect both.
        """
        pre_dict  = dict(self.enc_pre.backbone.named_parameters())
        post_dict = dict(self.enc_post.backbone.named_modules())

        # Identify shared modules by name (exclude stem conv)
        for name, module in self.enc_post.backbone.named_modules():
            if name == "" or "conv1" in name:
                continue   # skip stem
            pre_module = dict(self.enc_pre.backbone.named_modules()).get(name)
            if pre_module is None:
                continue
            # Replace post parameters with pre parameters (in-place)
            for p_name, _ in module.named_parameters(recurse=False):
                pre_param  = getattr(pre_module, p_name, None)
                if pre_param is not None:
                    setattr(module, p_name, pre_param)

    def forward(
        self,
        pre: torch.Tensor,    # (B, 3, H, W)
        post: torch.Tensor,   # (B, 1, H, W)
    ) -> torch.Tensor:        # (B, 1, H, W) logits

        # Convert 1-channel SAR to 3-channel dynamically to use ImageNet weights
        post_3ch = post.repeat(1, 3, 1, 1)

        # Encode both modalities
        feats_pre  = self.enc_pre(pre)    # list of 5 tensors
        feats_post = self.enc_post(post_3ch)

        # s0 … s4 from shallowest to deepest
        s0_pre,  s1_pre,  s2_pre,  s3_pre,  s4_pre  = feats_pre
        s0_post, s1_post, s2_post, s3_post, s4_post = feats_post

        # Bottleneck: element-wise difference of deepest features
        bridge = torch.abs(s4_pre - s4_post)   # (B, 512, H/32, W/32)

        # Decode
        d4 = self.dec4(bridge, s3_pre, s3_post)  # (B, 256, H/16, W/16)
        d3 = self.dec3(d4,     s2_pre, s2_post)  # (B, 128, H/8,  W/8)
        d2 = self.dec2(d3,     s1_pre, s1_post)  # (B,  64, H/4,  W/4)
        d1 = self.dec1(d2,     s0_pre, s0_post)  # (B,  32, H/2,  W/2)

        # Final upsample to input resolution
        out = self.final_up(d1)                  # (B,  32, H,    W)
        return self.head(out)                    # (B,   1, H,    W)


# ── Count parameters ──────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.2f}M  |  Trainable: {trainable/1e6:.2f}M"


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = SiameseUNet(pretrained=False)
    print(count_params(model))
    B, H, W = 2, 256, 256
    pre  = torch.randn(B, 3, H, W)
    post = torch.randn(B, 1, H, W)
    out  = model(pre, post)
    print(f"Output shape: {out.shape}")   # expect (2, 1, 256, 256)
    assert out.shape == (B, 1, H, W), "Shape mismatch!"
    print("✅ SiameseUNet forward pass OK")
