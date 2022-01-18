from typing import Tuple, Optional, List
from torch_geometric.typing import OptTensor, Adj, SparseTensor
from torch_geometric.nn import knn_graph

import torch
from torch import Tensor
import torch.nn as nn
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import reset

"""
E(n) Equivariant Graph Neural Networks, Victor Garcia Satorras, Emiel Hogeboom, Max Welling
https://arxiv.org/abs/2102.09844

Aadapted from https://github.com/vgsatorras/egnn/blob/main/models/egnn_clean/egnn_clean.py
to match into the Pytorch-Geometric MessagePassing framework
"""


def exists(val):
    return val is not None


def get_fully_connected_edges(n_nodes: int, add_self_loops: bool = False) -> Tensor:
    """
    Creates the edge_index in COO format in a fully-connected graph with :obj:`n_nodes` nodes.
    """
    edge_index = torch.cartesian_prod(torch.arange(n_nodes), torch.arange(n_nodes)).T
    if not add_self_loops:
        edge_index = edge_index.t()[edge_index[0] != edge_index[1]].t()

    return edge_index


def get_fully_connected_edges_in_batch(
    batch_num_nodes: Tensor,
    ptr: Tensor,
    edge_index: Tensor,
    add_self_loops: bool = False,
    edge_attr: OptTensor = None,
) -> Tuple[Tensor, OptTensor]:
    """
    Creates the edge_index in COO format in a fully-connected graph in a batch
    """
    fc_edge_index = [
        get_fully_connected_edges(int(n), add_self_loops) + int(p)
        for n, p in zip(batch_num_nodes, ptr)
    ]
    fc_edge_index = torch.cat(fc_edge_index, dim=-1).to(edge_index.device)

    # create dictionary with string as keys, e.g. [0, 1] meaning the connectivity between source node_id 0 to 1
    # the value of the position along dim=1 of edge_index
    source_target_to_edge_idx = {
        str([int(s), int(t)]): i
        for s, t, i in zip(edge_index[0], edge_index[1], range(edge_index.size(1)))
    }
    # positions of fake edge_index
    source_target_to_fc_edge_idx = {
        str([int(s), int(t)]): i
        for s, t, i in zip(
            fc_edge_index[0], fc_edge_index[1], range(fc_edge_index.size(1))
        )
    }

    fake_edges = [
        s
        for s in source_target_to_fc_edge_idx.keys()
        if s not in source_target_to_edge_idx.keys()
    ]
    fake_edges_ids = [source_target_to_fc_edge_idx[k] for k in fake_edges]
    # E_fc = fc_edge_index.shape[1]
    # E = edge_index.shape[1]
    # assert len(fake_edges) == E_fc - E
    fake_edge_index = fc_edge_index.t()[fake_edges_ids].t()

    # concatenate behind the fake-edges along the true edge_index
    all_edge_index = (
        torch.cat([edge_index, fake_edge_index], dim=-1).long().to(edge_index.device)
    )

    if edge_attr is not None:
        # concatenate/zero-pad the fake edge_attr behind the true edge_attr
        fake_edge_attr = torch.zeros(
            size=(fake_edge_index.size(1), edge_attr.size(-1)), device=edge_index.device
        )
        all_edge_attr = torch.cat([edge_attr, fake_edge_attr], dim=0)
    else:
        all_edge_attr = None

    return all_edge_index, all_edge_attr


class CoorsNorm(nn.Module):
    def __init__(self, eps: float = 1e-8, scale_init: float = 1.0):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class MyIdentity(nn.Module):
    def __init__(self):
        super(MyIdentity, self).__init__()

    def forward(self, x: Tensor, batch: OptTensor = None) -> Tensor:
        return x


# try replicate:
# https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/egnn_pytorch.py#L148
# PyG https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/egnn_pytorch_geometric.py also exists.


class E_GCL(MessagePassing):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim_expansion: int = 2,
        msg_channels: int = 16,
        use_fc: bool = False,
        use_velocity: bool = False,
        edge_channels: int = 0,
        use_coors_norm: bool = False,
        coors_norm_scale_init: float = 0.01,
        coor_weights_clamp_value: Optional[float] = None,
        use_node_norm: bool = False,
        dropout: float = 0.0,
        act_fn: nn.Module = nn.SiLU(),
        bias: bool = True,
        attention: bool = False,
        coords_agg: str = "add",
        aggr: str = "add",
        init_eps: float = 1e-3,
        tanh: bool = False,
        **kwargs,
    ):
        super(E_GCL, self).__init__(aggr=aggr, **kwargs)
        self.attention = attention
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.use_vel = use_velocity
        self.use_fc = use_fc
        self.init_eps = init_eps

        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.use_coors_norm = use_coors_norm
        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.coors_norm_scale_init = coors_norm_scale_init
        self.use_node_norm = use_node_norm

        # normalization layers
        self.node_norm = LayerNorm(in_channels) if use_node_norm else MyIdentity()
        self.coors_norm = (
            CoorsNorm(scale_init=coors_norm_scale_init)
            if use_coors_norm
            else MyIdentity()
        )

        edge_input_dim = 2 * in_channels + 1 + edge_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * dim_expansion, bias=bias),
            dropout,
            act_fn,
            nn.Linear(edge_input_dim * dim_expansion, msg_channels, bias=bias),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(
                in_channels + msg_channels, dim_expansion * in_channels, bias=bias
            ),
            dropout,
            act_fn,
            nn.Linear(dim_expansion * in_channels, out_channels, bias=bias),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(msg_channels, 2 * dim_expansion * msg_channels, bias=bias),
            dropout,
            act_fn,
            nn.Linear(2 * dim_expansion * msg_channels, 1, bias=bias),
        )

        if use_velocity:
            self.vel_mlp = nn.Sequential(
                nn.Linear(msg_channels, 2 * dim_expansion * msg_channels, bias=bias),
                dropout,
                act_fn,
                nn.Linear(2 * dim_expansion * msg_channels, 1, bias=bias),
            )
        else:
            self.vel_mlp = None

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(msg_channels, 1, bias=bias), nn.Sigmoid()
            )
        else:
            self.att_mlp = None

        self.reset_parameters()

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.normal_(module.weight, std=self.init_eps)

    # https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/egnn_pytorch.py#L219

    def reset_parameters(self):
        self.apply(self.init_)
        # reset(self.edge_mlp)
        # reset(self.node_mlp)
        # reset(self.coord_mlp)
        # torch.nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)
        # reset(self.vel_mlp)
        # reset(self.att_mlp)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Adj,
        vel: OptTensor = None,
        edge_attr: OptTensor = None,
        fc_edge_index: OptTensor = None,
        batch: OptTensor = None,
        batch_num_nodes: OptTensor = None,
        ptr: OptTensor = None,
    ) -> Tuple[Tensor, Tuple[Tensor, OptTensor]]:
        """"""

        if isinstance(edge_index, SparseTensor):
            row, col, _ = edge_index.coo()
            edge_index = torch.stack([row, col], dim=0)

        # if fc_edge_index and batch_num_nodes are not provided:
        # create fully-connected edge-index if not passed
        # note, then it assumes that the entire batch just consists of ONE graph
        if batch is None:
            batch = torch.zeros([x.size(0)], dtype=torch.long, device=x.device)
        if batch_num_nodes is None:
            batch_num_nodes = torch.tensor(
                [pos.size(0)], device=pos.device, dtype=torch.long
            )
        if ptr is None:
            ptr = torch.tensor([0], device=pos.device, dtype=torch.long)
        if fc_edge_index is None and self.use_fc:
            fc_edge_index, edge_attr = get_fully_connected_edges_in_batch(
                batch_num_nodes=batch_num_nodes,
                ptr=ptr,
                edge_index=edge_index,
                add_self_loops=False,
                edge_attr=edge_attr,
            )
        else:
            fc_edge_index = edge_index

        # propagate_type: (x: OptTensor, pos: Tensor, edge_attr: OptTensor) -> Tuple[Tensor,Tensor] # noqa
        out_x, out_pos = self.propagate(
            edge_index=fc_edge_index,
            orig_index=edge_index[1],
            x=x,
            pos=pos,
            edge_attr=edge_attr,
            size=None,
        )

        out_x = torch.cat([self.node_norm(x, batch), out_x], dim=1)
        out_x = self.node_mlp(out_x)

        if vel is None:
            out_pos += pos
            out_vel = None
        else:
            out_vel = self.vel_nn(x) * vel + out_pos
            out_pos = pos + out_vel

        return (out_x, (out_pos, out_vel))

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        pos_i: Tensor,
        pos_j: Tensor,
        edge_attr: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:

        msg = torch.sum((pos_i - pos_j).square(), dim=1, keepdim=True)
        msg = torch.cat([x_i, x_j, msg], dim=1)
        msg = msg if edge_attr is None else torch.cat([msg, edge_attr], dim=1)
        msg = self.edge_mlp(msg)
        if self.attention:
            msg = msg * self.att_mlp(msg)

        coors_weights = self.coord_mlp(msg)
        if self.coor_weights_clamp_value:
            coors_weights.clamp_(
                -self.coor_weights_clamp_value, self.coor_weights_clamp_value
            )
        pos_msg = self.coors_norm(pos_i - pos_j) * coors_weights

        return (msg, pos_msg)

    def aggregate(
        self, inputs: Tuple[Tensor, Tensor], index: Tensor, orig_index: Tensor
    ) -> Tuple[Tensor, Tensor]:
        node_aggr_messages = scatter(
            inputs[0][: len(orig_index)], orig_index, 0, reduce=self.aggr
        )
        node_aggr_positional = scatter(inputs[1], index, 0, reduce=self.coords_agg)
        return (node_aggr_messages, node_aggr_positional)

    def update(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        return inputs


class IntegerEncoder(nn.Module):
    def __init__(self, num_atoms: int, out_channels: int):
        super(IntegerEncoder, self).__init__()
        self.encoder = nn.Embedding(
            num_embeddings=num_atoms, embedding_dim=out_channels
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        # torch.nn.init.kaiming_uniform_(self.encoder.weight.data)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class Prot_EGNN(nn.Module):
    def __init__(
        self,
        depth: int = 3,
        dim: int = 256,
        msg_dim: int = 16,
        dropout: float = 0.0,
        residual: bool = True,
        num_nearest_neighbors: int = 32,
        num_atom_tokens: Optional[int] = None,
        num_edge_tokens: Optional[int] = None,
        dim_expansion: int = 2,
        use_fc: bool = False,
        edge_dim: int = 0,
        use_activation: bool = False,
        **kwargs,
    ):
        super(Prot_EGNN, self).__init__()

        self.depth = depth
        self.dim = dim
        self.residual = residual
        self.num_nearest_neighbors = num_nearest_neighbors
        self.dim_expansion = dim_expansion
        self.num_atom_tokens = num_atom_tokens
        self.num_edge_tokens = num_edge_tokens
        self.edge_dim = edge_dim
        self.use_fc = use_fc
        self.dropout = nn.Dropout(dropout)
        self.use_activation = use_activation
        self.activation_fnc = nn.SiLU()

        self.node_encoder = (
            IntegerEncoder(num_atom_tokens, dim) if exists(num_atom_tokens) else None
        )
        self.edge_encoder = (
            IntegerEncoder(num_edge_tokens, edge_dim)
            if exists(num_edge_tokens)
            else None
        )

        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                E_GCL(
                    in_channels=dim,
                    dim_expansion=dim_expansion,
                    msg_channels=msg_dim,
                    out_channels=dim,
                    edge_channels=edge_dim,
                    use_fc=use_fc,
                    **kwargs,
                )
            )

        self.reset_parameters()

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        if self.edge_encoder is not None:
            self.edge_encoder.reset_parameters()

        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        feats: Tensor,
        coors: Tensor,
        batch: OptTensor = None,
        ptr: OptTensor = None,
        edge_attr: OptTensor = None,
        vel: OptTensor = None,
        return_coor_changes: bool = False,
    ) -> Tuple[Tensor, Tensor, List]:

        # create kNN graph
        edge_index = knn_graph(coors.float(), k=self.num_nearest_neighbors, batch=batch)

        if batch is None:
            batch = torch.zeros([feats.size(0)], dtype=torch.long, device=feats.device)

        if ptr is None:
            ptr = torch.tensor([0], device=feats.device, dtype=torch.long)

        batch_num_nodes = torch.bincount(batch)

        # get fully-connected edge_index
        if self.use_fc:
            fc_edge_index, edge_attr = get_fully_connected_edges_in_batch(
                batch_num_nodes=batch_num_nodes,
                ptr=ptr,
                edge_index=edge_index,
                add_self_loops=False,
                edge_attr=edge_attr,
            )

        in_feats = self.node_encoder(feats.int())
        in_coors = coors
        if exists(edge_attr) and exists(self.edge_encoder):
            edge_attr = self.edge_encoder(edge_attr)

        coor_changes = []

        if return_coor_changes:
            coor_changes.append(coors)

        # iterate over layers
        for i, egnn_conv in enumerate(self.layers):
            if self.use_fc:
                feats, coors = egnn_conv(
                    x=in_feats,
                    pos=in_coors,
                    edge_index=edge_index,
                    fc_edge_index=fc_edge_index,
                    batch=batch,
                    edge_attr=edge_attr,
                    vel=vel,
                )
            else:
                feats, coors = egnn_conv(
                    x=in_feats,
                    pos=in_coors,
                    edge_index=edge_index,
                    fc_edge_index=edge_index,
                    batch=batch,
                    edge_attr=edge_attr,
                    vel=vel,
                )

            coors = coors[0]  # coors is a tuple of coors and velocity (None)

            # use activation for feats ?
            if self.use_activation:
                if i < len(self.layers) - 1:
                    feats = self.activation_fnc(feats)

            # residual connections
            if self.residual:
                feats += in_feats
                coors += in_coors
            in_feats = feats
            in_coors = coors

            if return_coor_changes:
                coor_changes.append(coors)

        return feats, coors, coor_changes


class DownstreamAttentiveFFN(torch.nn.Module):
    """
    A 1-layer MLP that utilizes an attentive set-aggregation network wrt. the sample in the multi-set
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super(DownstreamAttentiveFFN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.activation = nn.SiLU()
        self.fc1 = nn.Linear(in_channels, in_channels // 4, bias=bias)
        self.attention = nn.Linear(in_channels // 4, 1, bias=bias)
        self.out_layer = nn.Linear(in_channels // 4, out_channels, bias=bias)

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        x = self.dropout(self.activation(self.fc1(x)))
        attention_weights = self.attention(x)
        attention_weights = scatter_softmax(src=attention_weights, index=index, dim=0)

        x = attention_weights.view(-1, 1) * x
        x = scatter(src=x, index=index, dim=0, reduce="add")
        x = self.out_layer(x)
        return x


if __name__ == "__main__":
    model = Prot_EGNN(
        num_atom_tokens=18,
        depth=3,
        dim=256,
        msg_dim=16,
        dim_expansion=2,
        num_nearest_neighbors=32,
        use_coors_norm=True,
        coor_weights_clamp_value=2.0,
        use_fc=False,
        dropout=0.0,
        residual=True,
    )

    params = sum(m.numel() for m in model.parameters() if m.requires_grad)

    print(f"Model consists of {params} trainable parameters.")
    # Model consists of 2452770 trainable parameters.
