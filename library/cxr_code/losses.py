import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights
    def forward(self, preds, targets):
        loss = 0
        for i,c in enumerate(self.losses):
            loss += self.weights[i]*c(preds, targets)
        return loss

class BCELogLoss(nn.Module):
    def __init__(self, device, smooth=False):
        super(BCELogLoss, self).__init__()
        self.device = device
        # label smoothing
        self.smooth = smooth
    def forward(self, output, target):
        # Label smoothing
        if self.smooth:
            eps = np.random.uniform(0.01, 0.05)
            target = (1-eps)*target + eps / target.size()[1]
        return F.binary_cross_entropy_with_logits(output, target)

class ReversedBCELoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = torch.clamp(preds, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(targets, min=1e-4, max=1.0)
        rce = - torch.sum(preds * torch.log(label_one_hot), dim=1).mean()
        return rce

class NormalizedBCELoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, preds, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        preds = torch.sigmoid(preds)
        denorm = -torch.log(preds*(1-preds)) # log(a) + log(b) = log(ab)
        return (bce/denorm).mean()

class SymetricBCELoss(nn.Module):
    def __init__(self, alpha, beta, device):
        super(SymetricBCELoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # CCE
        bce = self.bce(preds, targets)

        # RCE
        preds = torch.sigmoid(preds)
        preds = torch.clamp(preds, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(targets, min=1e-4, max=1.0)
        rce = - torch.sum(preds * torch.log(label_one_hot), dim=1).mean()

        # Loss
        loss = self.alpha * bce + self.beta * rce
        return loss