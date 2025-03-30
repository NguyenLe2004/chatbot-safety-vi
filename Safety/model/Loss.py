import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=3.0, reduction='mean'):
        """
        Focal Loss for Multi-Label Classification
        Args:
            alpha: Weighting factor to address class imbalance (float or tensor).
            gamma: Focusing parameter for hard examples.
            reduction: Reduction method (mean, sum, or none).
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Model outputs (before sigmoid), shape (batch_size, num_classes).
            targets: Ground truth labels (0 or 1), shape (batch_size, num_classes).
        Returns:
            Loss value (scalar or tensor).
        """
        probs = torch.sigmoid(logits)  
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Apply reduction (mean or sum)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
