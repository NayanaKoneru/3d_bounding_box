import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast


from src.dataset import BBoxDataset
from src.transforms import get_train_transforms
from src.model import get_model
from src.losses import BoxLoss
from src.utils import save_checkpoint, set_seed




def train(cfg):
    set_seed(cfg['seed'])
    # load items (this example expects pre-parsed items list)
    train_items = ... # implement loader e.g. from csv/json
    val_items = ...


    train_ds = BBoxDataset(train_items, transforms=get_train_transforms(cfg['image_size']))
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, collate_fn=lambda x: x)

    model = get_model(backbone_name=cfg.get('backbone','resnet50'), pretrained=True, out_dim=7, use_transformer=cfg.get('use_transformer', False))
    model = model.to(cfg['device'])

    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    criterion = BoxLoss()

    scaler = GradScaler()

    best_metric = -1
    for epoch in range(cfg['epochs']):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # batch is list of samples because collate_fn above
            imgs = torch.stack([b[0] for b in batch]).to(cfg['device'])
            targets = torch.stack([b[1]['boxes'].flatten() for b in batch]).to(cfg['device']) # user to adapt


            optimizer.zero_grad()
            with autocast():
                preds = model(imgs)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss/ (pbar.n+1))


        scheduler.step()
        # validation (call evaluate)
        # save checkpoint
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(cfg['checkpoint_dir'], f'ckpt_{epoch}.pth'))