import torch
import os
from pytorch_lightning.core.lightning import LightningModule
from egnn_pytorch import EGNN_Network


class EGNNLayer(LightningModule):
    def __init__(self):
        super().__init__()

        self.egnn = EGNN_Network(
            num_tokens=18,
            dim=64,
            depth=3,
            num_nearest_neighbors=32,
            norm_coors=True,
            coor_weights_clamp_value=2.0,
        )

    def forward(self, coords, feats):
        feats = feats.int().to(self.device)
        coords = coords.float().to(self.device)
        mask = torch.ones_like(feats).bool().to(self.device)

        feats_out, _ = self.egnn(feats, coords, mask=mask)
        return feats_out


model = EGNNLayer()
curdir = os.path.dirname(os.path.realpath(__file__))
model.load_state_dict(torch.load(f"{curdir}/pege.pth"))
