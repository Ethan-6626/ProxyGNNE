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


def partition_graph(data, nparts):
    n = data.num_nodes
    adjacency = [[] for _ in range(n)]
    edge_index = data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        adjacency[src].append(dst)
        adjacency[dst].append(src)  
    _, parts = pymetis.part_graph(nparts, adjacency=adjacency)
    return np.array(parts)


def extract_partition_subgraph(data, parts, partition_id):
    from torch_geometric.data import Data

    idx = np.where(parts == partition_id)[0]

    if len(idx) == 0:
        raise ValueError(f"{partition_id} Can not find node.")

    x_sub = data.x[idx]
    y_sub = data.y[idx]

    idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(idx)}

    edge_index_cpu = data.edge_index.cpu().numpy()
    mask = np.isin(edge_index_cpu[0], idx) & np.isin(edge_index_cpu[1], idx)
    edge_index_sub = data.edge_index[:, mask]

    edge_index_sub = edge_index_sub.cpu().numpy()
    edge_index_sub[0] = [idx_mapping[src] for src in edge_index_sub[0]]
    edge_index_sub[1] = [idx_mapping[dst] for dst in edge_index_sub[1]]
    edge_index_sub = torch.tensor(edge_index_sub, dtype=torch.long)


    subgraph_data = Data(x=x_sub, edge_index=edge_index_sub, y=y_sub)
    subgraph_data.num_classes = data.num_classes if hasattr(data, 'num_classes') else data.y.max().item() + 1
    subgraph_data.n_id = torch.tensor(idx, dtype=torch.long)

    return subgraph_data