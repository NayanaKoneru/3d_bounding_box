import os
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path


def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(state: dict, path: str):
    """Save model + optimizer checkpoint."""
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def load_checkpoint(model, ckpt_path: str, optimizer=None):
    """
    Load checkpoint into model (and optimizer if provided).
    Returns the saved epoch if available.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt.get("epoch", None)


def get_timestamp():
    """Return YYYYMMDD_HHMMSS string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def plot_boxes_cv2(img, boxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image using cv2.
    img: numpy array (H,W,3)
    boxes: Nx4 tensor/list of [x1,y1,x2,y2]
    """
    import cv2

    img = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def collate_fn(batch):
    """
    Custom collate function for variable number of boxes.
    Returns:
      images: tensor BxCxHxW
      targets: list of dicts
    """
    imgs = []
    targets = []
    for img, target in batch:
        imgs.append(img)
        targets.append(target)

    imgs = torch.stack(imgs, dim=0)
    return imgs, targets