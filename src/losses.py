import torch
import torch.nn as nn


class BoxLoss(nn.Module):
    def __init__(self, l1_weight=1.0, iou_weight=1.0):
        super().__init__()
        self.l1 = nn.SmoothL1Loss()
        self.l1_weight = l1_weight
        self.iou_weight = iou_weight


    def forward(self, pred, target):
        # pred: [B, 7], target: [B,7]
        l1_loss = self.l1(pred, target)
        # placeholder IoU term (user should replace with real IoU for 3D boxes)
        iou_loss = torch.tensor(0., device=pred.device)
        return self.l1_weight*l1_loss + self.iou_weight*iou_loss