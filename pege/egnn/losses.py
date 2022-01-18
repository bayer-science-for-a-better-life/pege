import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add, scatter_mean


def beta_nll_loss(y_pred: Tensor, y_true: Tensor, eps: float = 1e-6) -> Tensor:
    """
    A functional method to calculate the negative log-likelihood for the beta regression
    :param y_pred: [*, 2*51] tensor where the first 51 column state the mean and the last 51 the precision parameter
    :param y_true: [*, 51] tensor of probability values to predict
    :param eps: defaults to 1e-6 for numerical stability
    :return: negative log-likelihood
    """
    mu, phi = y_pred.chunk(2, dim=-1)
    # mu is in real space, apply inverse link function to map to [0,1]
    mu = mu.sigmoid()  # mean parameter
    # phi is before applying the inverse link function in real space, but as phi represent precision, map to R+
    # phi = phi.exp()  # precision parameter
    phi = F.softplus(phi)

    y_true = y_true.clamp(eps, 1 - eps)
    loglik = torch.lgamma(phi + eps) - torch.lgamma(phi * mu + eps) \
             - torch.lgamma((1 - mu) * phi + eps) + (mu * phi) * torch.log(y_true) \
             + ((1 - mu) * phi - 1) * torch.log(1 - y_true)

    return -loglik


def gaussian_nll_loss(y_pred: Tensor, y_true: Tensor, eps: float = 1e-6):
    """
    A functional method to calculate the negative log-likelihood for the gaussian regression.
    :param y_pred: [*, 2*out_dim]
                   tensor where the first out_dim column state the mean and the last out_dim the precision logvar
    :param y_true: [*, out_dim] tensor of target values to predict
    :param eps: defaults to 1e-6 for numerical stability
    :return: negative log-likelihood
    """
    mu, logvar = y_pred.chunk(2, dim=-1)
    logvar += eps
    loss = 0.5 * (logvar + (y_true - mu) ** 2 / torch.exp(logvar)).view(y_pred.size(0), -1)
    # sum ?
    loss = loss.sum(dim=-1)
    return loss


class MultiTaskLoss(torch.nn.Module):
    """
    A Super-Method to implement frequentist and probabilistic multitask lossfunction
    """
    def __init__(self,
                 single_reduction: str = "mean",
                 batch_reduction: str = "mean",
                 ):
        super(MultiTaskLoss, self).__init__()

        assert single_reduction in ["sum", "mean"]
        assert batch_reduction in ["sum", "mean"]
        self.single_reduction = single_reduction
        self.batch_reduction = batch_reduction
        self.reduction_fnc = scatter_mean if single_reduction == "mean" else scatter_add

    def calc_hs_loss(self, y_hs_pred: Tensor, y_hs_true: Tensor) -> Tensor:
        raise NotImplementedError("implement this method")

    def calc_pks_loss(self, y_pks_pred: Tensor, y_pks_true: Tensor) -> Tensor:
        raise NotImplementedError("implement this method")

    def calc_pi_loss(self, y_pI_pred: Tensor, y_pI_true: Tensor) -> Tensor:
        raise NotImplementedError("implement this method")

    def forward(self,
                y_hs_true: Tensor, y_hs_pred: Tensor, y_hs_batch: Tensor,
                y_pks_true: Tensor, y_pks_pred: Tensor, y_pks_batch: Tensor,
                y_pI_true: Tensor, y_pI_pred: Tensor) -> Tensor:


        # Calculate loss for hydrogen probabilities
        loss_hs = self.calc_hs_loss(y_hs_pred=y_hs_pred, y_hs_true=y_hs_true)  # [batch_num_hydrogens, 51]
        if self.single_reduction == "mean":
            loss_hs = loss_hs.mean(dim=-1)
        else:
            loss_hs = loss_hs.sum(dim=-1)

        # in: [batch_num_hydrogens, ]
        loss_hs = self.reduction_fnc(loss_hs, y_hs_batch, dim=0)  # [batch_size,]        
        #loss_hs = torch.zeros(max(y_pks_batch)+1).to('cuda:5')

        ## Calculate loss for pks values
        loss_pks = self.calc_pks_loss(y_pks_pred=y_pks_pred, y_pks_true=y_pks_true)  # [batch_num_heavyAtomsHs,]
        loss_pks = self.reduction_fnc(loss_pks, y_pks_batch, dim=0)  # [batch_size,]

        ## Calculate loss for pI value(s)
        loss_pI = self.calc_pi_loss(y_pI_pred=y_pI_pred, y_pI_true=y_pI_true)  # [batch_size,]                

        ## combine all losses
        multitask_loss = torch.stack([loss_hs, loss_pks, loss_pI], dim=-1)  # [batch_size, 3]
        # multitask_loss = loss_hs  # [batch_size, ]
        
        # reduce batch
        if self.batch_reduction == "mean":
            multitask_loss = multitask_loss.mean(dim=0)
        else:
            multitask_loss = multitask_loss.sum(dim=0)

        return multitask_loss, torch.zeros(1), torch.zeros(1)


class FrequentistMultitaskLoss(MultiTaskLoss):
    def __init__(self,
                 single_reduction: str = "mean",
                 batch_reduction: str = "mean",
                 h_mode: str = "classification",
                 metric: str = "l2"):
        super(FrequentistMultitaskLoss, self).__init__(single_reduction=single_reduction,
                                                       batch_reduction=batch_reduction)

        assert metric in ["l1", "l2"]
        assert h_mode in ["regression", "classification"]
        self.h_mode = h_mode
        self.loss_fnc = nn.MSELoss(reduction="none") if metric == "l2" else nn.L1Loss(reduction="none")

    def calc_hs_loss(self, y_hs_pred: Tensor, y_hs_true: Tensor) -> Tensor:
        if self.h_mode == "regression":
            return self.loss_fnc(input=y_hs_pred.sigmoid(), target=y_hs_true)
        else:
            return F.binary_cross_entropy_with_logits(input=y_hs_pred, target=y_hs_true, reduction="none")

    def calc_pks_loss(self, y_pks_pred: Tensor, y_pks_true: Tensor) -> Tensor:
        return self.loss_fnc(input=y_pks_pred, target=y_pks_true)

    def calc_pi_loss(self, y_pI_pred: Tensor, y_pI_true: Tensor) -> Tensor:
        return self.loss_fnc(input=y_pI_pred, target=y_pI_true)


class ProbabilisticMultitaskLoss(MultiTaskLoss):
    def __init__(self,
                 single_reduction: str = "mean",
                 batch_reduction: str = "mean",
                 eps: float = 1e-6,
                 **kwargs):
        super(ProbabilisticMultitaskLoss, self).__init__(single_reduction=single_reduction,
                                                         batch_reduction=batch_reduction)
        self.eps = eps

    # Beta regression loss
    def calc_hs_loss(self, y_hs_pred: Tensor, y_hs_true: Tensor) -> Tensor:
        return beta_nll_loss(y_pred=y_hs_pred, y_true=y_hs_true, eps=self.eps)

    # Gaussian regression loss
    def calc_pks_loss(self, y_pks_pred: Tensor, y_pks_true: Tensor) -> Tensor:
        return gaussian_nll_loss(y_pred=y_pks_pred, y_true=y_pks_true, eps=self.eps)

    # Gaussian regression loss
    def calc_pi_loss(self, y_pI_pred: Tensor, y_pI_true: Tensor) -> Tensor:
        return gaussian_nll_loss(y_pred=y_pI_pred, y_true=y_pI_true, eps=self.eps)


# old

class ScaledMSE(nn.Module):
    def __init__(self):
        super(ScaledMSE, self).__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        diff = abs(y_true - y_pred)
        max_diff = 1 - ((diff <= 0.5) * diff + (diff > 0.5) * diff)
        mse_scaled = (1 - (max_diff - diff)) ** 2

        return torch.mean(mse_scaled)
