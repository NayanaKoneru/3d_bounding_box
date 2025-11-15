import os
import cv2
import torch
from torch.utils.data import Dataset


class BBoxDataset(Dataset):
    def __init__(self, items, transforms=None):
        # items: list of dicts {'img_path':..., 'bboxes':[[x1,y1,x2,y2]], 'labels':[...]}
        self.items = items
        self.transforms = transforms


    def __len__(self):
        return len(self.items)


    def __getitem__(self, idx):
        it = self.items[idx]
        img = cv2.imread(it['img_path'])[:, :, ::-1]
        bboxes = it.get('bboxes', [])
        labels = it.get('labels', [0]*len(bboxes))


        if self.transforms:
            transformed = self.transforms(image=img, bboxes=bboxes, labels=labels)
            img = transformed['image']
            bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32) if len(transformed['bboxes'])>0 else torch.zeros((0,4))
            labels = torch.tensor(transformed['labels'], dtype=torch.long)
        else:
            img = torch.tensor(img).permute(2,0,1).float()/255.
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)


        target = {'boxes': bboxes, 'labels': labels}
        return img, target