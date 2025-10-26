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
from graph_sampling import sample_graph_saint_subgraph
from graph_partition import partition_graph
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def combine_subgraphs(data, subgraph1, subgraph2, device='cuda'):
    data = data.to(device)
    subgraph1 = subgraph1.to(device)
    subgraph2 = subgraph2.to(device)

    n_id1 = subgraph1.n_id if hasattr(subgraph1, 'n_id') else torch.unique(subgraph1.edge_index)
    n_id2 = subgraph2.n_id if hasattr(subgraph2, 'n_id') else torch.unique(subgraph2.edge_index)
    n_id1 = n_id1.to(device)
    n_id2 = n_id2.to(device)

    n_id1_set = set(n_id1.cpu().numpy())
    n_id2_set = set(n_id2.cpu().numpy())
    overlap_nodes = n_id1_set.intersection(n_id2_set)
    combined_n_id = torch.unique(torch.cat([n_id1, n_id2]), sorted=True).to(device)

    node_map = {int(old): new for new, old in enumerate(combined_n_id.cpu().numpy())}

    x_combined = data.x[combined_n_id]
    y_combined = data.y[combined_n_id]

    def remap_edges_with_connections(subgraph, node_map, orig_data):
        if hasattr(subgraph, 'n_id'):
            local_nodes = set(subgraph.n_id.cpu().numpy())
        else:
            local_nodes = set(torch.unique(subgraph.edge_index).cpu().numpy())

        edge_list = []

        edge_index = subgraph.edge_index.cpu()
        for i in range(edge_index.size(1)):
            src, dst = int(edge_index[0, i]), int(edge_index[1, i])
            if src in node_map and dst in node_map:
                edge_list.append([node_map[src], node_map[dst]])

        orig_edge_index = orig_data.edge_index.cpu()
        for i in range(orig_edge_index.size(1)):
            src, dst = int(orig_edge_index[0, i]), int(orig_edge_index[1, i])
            if src in node_map and dst in node_map:
                if src in local_nodes or dst in local_nodes:
                    edge_list.append([node_map[src], node_map[dst]])

        return torch.tensor(edge_list, dtype=torch.long, device=device).t()

    edge_index1 = remap_edges_with_connections(subgraph1, node_map, data)
    edge_index2 = remap_edges_with_connections(subgraph2, node_map, data)

    edge_index_combined = torch.cat([edge_index1, edge_index2], dim=1)
    edge_index_combined = torch.unique(edge_index_combined, dim=1)

    if len(overlap_nodes) > 0:
        cross_edges = []
        for node in overlap_nodes:
            mask = (data.edge_index[0] == node) | (data.edge_index[1] == node)
            connected_edges = data.edge_index[:, mask]
            for i in range(connected_edges.size(1)):
                src, dst = int(connected_edges[0, i]), int(connected_edges[1, i])
                if src in node_map and dst in node_map:
                    cross_edges.append([node_map[src], node_map[dst]])

        if cross_edges:
            cross_edges = torch.tensor(cross_edges, dtype=torch.long, device=device).t()
            edge_index_combined = torch.cat([edge_index_combined, cross_edges], dim=1)
            edge_index_combined = torch.unique(edge_index_combined, dim=1)
    data_combined = Data(
        x=x_combined.to(device),
        edge_index=edge_index_combined.to(device),
        y=y_combined.to(device),
        num_classes=getattr(data, 'num_classes', data.y.max().item() + 1),
        n_id=combined_n_id.to(device)
    )

    return data_combined

def load_and_describe_dataset(dataset_name,root,is_des=True):
    dataset=Planetoid(root=root,name=dataset_name)
    data=dataset[0]

    if is_des==True:
        print(f'{dataset.name}')
        print(f'{data.num_nodes}')
        print(f'{data.num_edges}')
        print(f'{dataset.num_classes}')
        print(f'{dataset.num_node_features}')
    return dataset,data

def extract_partition_subgraph(data, parts, partition_id):
    from torch_geometric.data import Data

    idx = np.where(parts == partition_id)[0]
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

def process_partitioned_and_sampled_graphs(data, best_nparts,device='cuda',**sampling_kwargs):  
    data = data.to(device)
    partition_Graph = partition_graph(data.cpu(), best_nparts)
    unique_parts = np.unique(partition_Graph)
    partition_id = unique_parts[0]
    subgraph_data = extract_partition_subgraph(data.cpu(), partition_Graph, partition_id)
    subgraph_data = subgraph_data.to(device)

    sampling_params = {
        'batch_size': 128,
        'walk_length': 4,
        'exclude_n_id': None,
        'device': device
    }
    sampling_params.update(sampling_kwargs)
    
    sampled_subgraph = sample_graph_saint_subgraph(
        subgraph_data,  # 对分区子图采样
        **sampling_params
    )

    combined_subgraph = combine_subgraphs(
        data, 
        subgraph_data, 
        sampled_subgraph, 
        device=device
    )

    return combined_subgraph


def combined_graph(data, best_nparts, max_retries=3, device='cuda'):

    for attempt in range(max_retries):
        try:
            combined_subgraph = process_partitioned_and_sampled_graphs(
                data,
                best_nparts,
                device=device
            )
            if combined_subgraph is not None and combined_subgraph.num_nodes > 0:
                return combined_subgraph.to(device)
            else:
                if attempt == max_retries - 1:
                    return data.to(device)
        except Exception as e:
            if attempt == max_retries - 1:
                return data.to(device)
            continue

    return data.to(device)

