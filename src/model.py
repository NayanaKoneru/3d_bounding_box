import torch
import torch.nn as nn
import torchvision.models as models


class RegressionHead(nn.Module):
    def __init__(self, in_features, out_dim=7):
        super().__init__()
        self.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(512, out_dim)
        )


    def forward(self, x):
        return self.fc(x)


class BackboneWithHead(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, out_dim=7, use_transformer=False):
        super().__init__()
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        # remove classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1]) # global pool last
        feat_dim = backbone.fc.in_features
        self.head = RegressionHead(feat_dim, out_dim)


        # Optional: add small transformer encoder if use_transformer True
        if use_transformer:
            self.transformer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=8)
        else:
            self.transformer = None

    def forward(self, x):
        # x: [B, C, H, W]
        f = self.backbone(x) # [B, feat_dim, 1, 1]
        f = f.view(f.size(0), -1)
        if self.transformer is not None:
            # expand to sequence len=1 (toy example) - replace with proper patch embedding for full transformer
            seq = f.unsqueeze(1)
            seq = self.transformer(seq)
            f = seq.squeeze(1)
        out = self.head(f)
        return out

def get_model(**kwargs):
    return BackboneWithHead(**kwargs)