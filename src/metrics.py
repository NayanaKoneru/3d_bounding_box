import torch


def box_iou_2d(box1, box2):
    """
    Compute IoU between two 2D boxes.
    box = [x1, y1, x2, y2]
    """
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-6
    return inter / union


def compute_iou_metric(preds, targets):
    """
    Compute average IoU across all samples.
    preds: list of tensors (model outputs)
    targets: list of dicts containing {'boxes': tensor}
    """
    ious = []

    # flatten preds list -> tensor
    preds = torch.cat(preds, dim=0)

    for pred, tgt in zip(preds, targets):
        # this assumes each sample has exactly 1 box regression target
        if tgt["boxes"].numel() == 0:
            continue

        pred_box = pred[:4]  # x1,y1,x2,y2 — adjust for your task
        gt_box = tgt["boxes"].flatten()

        iou = box_iou_2d(pred_box, gt_box)
        ious.append(iou.item())

    if len(ious) == 0:
        return 0.0

    return sum(ious) / len(ious)