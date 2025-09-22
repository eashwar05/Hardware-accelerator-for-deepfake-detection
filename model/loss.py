import torch
import torch.nn as nn
import torch.nn.functional as F

class SFFNLoss(nn.Module):
    """Weighted sum of Classification Loss and Reconstruction Loss."""
    def __init__(self, lambda_cls=1.0, mu_recon=0.5):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.mu_recon = mu_recon
        self.classification_loss = nn.CrossEntropyLoss()
        self.reconstruction_loss = nn.L1Loss()

    def forward(self, model_outputs, targets):
        cls_output, recon_output = model_outputs
        cls_labels, recon_targets = targets
        loss_cls = self.classification_loss(cls_output, cls_labels)
        recon_targets_downsampled = F.interpolate(
            recon_targets, size=recon_output.shape[-2:], mode='bilinear', align_corners=False)
        loss_recon = self.reconstruction_loss(recon_output, recon_targets_downsampled)
        total_loss = (self.lambda_cls * loss_cls) + (self.mu_recon * loss_recon)
        return total_loss, loss_cls, loss_recon