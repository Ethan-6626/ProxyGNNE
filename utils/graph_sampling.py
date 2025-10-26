import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCNConv, global_mean_pool
import pymetis
import numpy as np
from torch_geometric.utils import to_networkx, k_hop_subgraph, subgraph
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import networkx as nx
import yaml
from tqdm import tqdm
from model import ModelWrapper, ProxyGNNModel
from torch_geometric.explain.metric import fidelity
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.data import Data
import time
import random


def sample_graph_saint_subgraph(data, exclude_n_id=None, batch_size=128, walk_length=4, device='cuda'):
    data = data.cpu()
    
    if exclude_n_id is not None:
        mask = torch.ones(data.num_nodes, dtype=torch.bool)
        mask[exclude_n_id] = False
        edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
        data_filtered = Data(
            x=data.x,
            edge_index=data.edge_index[:, edge_mask],
            y=data.y[mask]
        )
    else:
        data_filtered = data

    loader = GraphSAINTRandomWalkSampler(
        data_filtered,
        batch_size=batch_size,
        walk_length=walk_length,
        num_steps=1,
        sample_coverage=100
    )

    for sampled_data in loader:
        if not hasattr(sampled_data, 'n_id') or sampled_data.n_id is None:
            sampled_data.n_id = torch.unique(sampled_data.edge_index).to(torch.long)
        else:
            sampled_data.n_id = sampled_data.n_id.to(torch.long)
        return sampled_data.to(device)

    return None