import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from src.dataset import BBoxDataset
from src.transforms import get_valid_transforms
from src.metrics import compute_iou_metric




def evaluate(model, items, cfg):
    model.eval()
    ds = BBoxDataset(items, transforms=get_valid_transforms(cfg['image_size']))
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=4, collate_fn=lambda x: x)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            imgs = torch.stack([b[0] for b in batch]).to(cfg['device'])
            targets = [b[1] for b in batch]
            preds = model(imgs)