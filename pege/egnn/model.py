import pytorch_lightning as pl

import torch
from torch.nn import Sequential, Linear, SiLU

from typing import Tuple

# custom module
from pege.egnn.convs import Prot_EGNN
from pege.egnn.losses import FrequentistMultitaskLoss, ProbabilisticMultitaskLoss


class EGNNprobsPyG(pl.LightningModule):
    def __init__(
        self,
        dim: int = 256,
        depth: int = 3,
        msg_dim: int = 16,
        dim_expansion: int = 2,
        weights: list = [1 / 3, 1 / 3, 1 / 3],
        use_activation: bool = False,
        single_reduction: str = "mean",
        batch_reduction: str = "mean",
        h_mode: str = "classification",
        target_mode: str = "frequentist",
        use_coors_norm: bool = True,
        use_node_norm: bool = False,
        dropout: float = 0.0,
        residual: bool = True,
        num_nearest: int = 32,
        initial_learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        batch_size: int = 1,
        num_workers: int = 0,
        patience_epochs_lr: int = 0,
        max_epochs: int = 30,
        data_dir: str = "/home/giwru/eeg_data",
    ):
        super().__init__()
        assert h_mode in ["regression", "classification"]
        assert target_mode in ["frequentist", "probabilistic"]

        self.save_hyperparameters()
        self.data_dir = data_dir
        self.max_epochs = max_epochs
        self.lr = initial_learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patience_epochs_lr = patience_epochs_lr

        self.dim = dim
        self.depth = dim
        self.num_nearest = num_nearest

        self.weights = torch.tensor(weights)
        self.target_mode = target_mode

        self.gnn = Prot_EGNN(
            num_atom_tokens=18,
            dim=dim,
            depth=depth,
            msg_dim=msg_dim,
            dim_expansion=dim_expansion,
            num_nearest_neighbors=num_nearest,
            use_coors_norm=use_coors_norm,
            use_node_norm=use_node_norm,
            coor_weights_clamp_value=2.0,
            use_fc=False,
            dropout=dropout,
            residual=residual,
            use_activation=use_activation,
        )

        if target_mode == "frequentist":
            hs_out_dim = 53
        else:
            hs_out_dim = 53 * 2

        self.fc_hs = Sequential(
            Linear(dim, dim // 4, bias=True),
            SiLU(),
            Linear(dim // 4, hs_out_dim, bias=True),
        )

        regr_out = 1 if target_mode == "frequentist" else 2
        self.fc_pk = Sequential(
            Linear(dim, dim // 4, bias=True),
            SiLU(),
            Linear(dim // 4, regr_out, bias=True),
        )

        self.fc_pI = Sequential(
            Linear(dim, dim // 4, bias=True),
            SiLU(),
            Linear(dim // 4, regr_out, bias=True),
        )

        # for loss calculation and calculating gradients:
        if target_mode == "frequentist":
            self.loss_fnc = FrequentistMultitaskLoss(
                single_reduction=single_reduction,
                batch_reduction=batch_reduction,
                h_mode=h_mode,
                metric="l2",
            )
        else:
            self.loss_fnc = ProbabilisticMultitaskLoss(
                single_reduction=single_reduction,
                batch_reduction=batch_reduction,
                h_mode=h_mode,
            )

        # for metric calculation, use regression metrics as compared before:
        self.metric_l1_loss = FrequentistMultitaskLoss(
            single_reduction, batch_reduction, h_mode="regression", metric="l1"
        )
        self.metric_l2_loss = FrequentistMultitaskLoss(
            single_reduction, batch_reduction, h_mode="regression", metric="l2"
        )

    def forward(
        self, coors: torch.Tensor, feats: torch.Tensor, hs_i_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = torch.zeros_like(feats)
        ptr = torch.Tensor([0, len(feats)])

        # get node embeddings
        feats_out, _, _ = self.gnn(
            feats=feats, coors=coors, batch=batch, ptr=ptr, return_coor_changes=False
        )

        # NaN masking
        feats_out = torch.nan_to_num(input=feats_out, nan=0.0)

        # index H probs
        X_h = feats_out[hs_i_ids]
        # predict H probs
        y_hs_pred = self.fc_hs(X_h)
        y_hs_pred = y_hs_pred.sigmoid()

        return feats_out, y_hs_pred


import os

curdir = os.path.dirname(os.path.abspath(__file__))

model = EGNNprobsPyG()
model.eval()
model.load_state_dict(torch.load(f"{curdir}/final_model.pth"))
