import torch
import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, label_weight: torch.Tensor = None, pos_weight: torch.Tensor = None):
        super(BCEWithLogitsLoss, self).__init__()
        self.label_weight = label_weight
        self.pos_weight = pos_weight
        self.loss_fn = nn.BCEWithLogitsLoss(weight=self.label_weight, pos_weight=self.pos_weight)

    def forward(self, logits, labels, masks):
        loss = self.loss_fn(logits, labels)
        return loss
