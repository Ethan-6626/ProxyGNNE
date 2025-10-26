
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

def calculate_fidelity(model, data, node_idx):
    
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)

    wrapped_model = ModelWrapper(model)

    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        index=node_idx
    )

    fid_pos, fid_neg = fidelity(explainer, explanation)

    with torch.no_grad():
        original_logits = model(data)
    prob = F.softmax(original_logits[node_idx], dim=0).cpu().numpy()
    label = data.y[node_idx].item()
    return fid_pos, fid_neg, prob, label