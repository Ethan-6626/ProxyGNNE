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
from sklearn.model_selection import train_test_split

def train_model(model, data, optimizer, num_epochs=400, print_interval=100):
    model.train()  

    for epoch in range(1, num_epochs + 1):  
        optimizer.zero_grad()  
        out = model(data)  
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  
        loss.backward()  
        optimizer.step()  
        if epoch % print_interval == 0 or epoch == 1 or epoch == num_epochs:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    return model

def evaluate_model(model, data):
    model.eval()  
    with torch.no_grad():  
        out = model(data)  
        _, pred = out.max(dim=1)  
        correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())  
        acc = correct / int(data.test_mask.sum())  
    print(f'acc:{acc:.4f}')
    return acc

def evaluate_proxy_model(proxy_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proxy_data = proxy_data.to(device)

    num_nodes = proxy_data.num_nodes
    indices = torch.randperm(num_nodes)
    train_indices = indices[:int(0.8 * num_nodes)]
    val_indices = indices[int(0.8 * num_nodes):]

    proxy_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    proxy_data.train_mask[train_indices] = True
    proxy_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    proxy_data.val_mask[val_indices] = True
    num_classes = proxy_data.num_classes if hasattr(proxy_data, 'num_classes') else proxy_data.y.max().item() + 1
    proxy_model = ProxyGNNModel(num_node_features=proxy_data.num_features,
                                hidden_dim=16,
                                num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(proxy_model.parameters(), lr=0.01, weight_decay=5e-4)

    proxy_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = proxy_model(proxy_data)
        loss = F.nll_loss(out[proxy_data.train_mask], proxy_data.y[proxy_data.train_mask])
        loss.backward()
        optimizer.step()
    proxy_model.eval()
    with torch.no_grad():
        out = proxy_model(proxy_data)
        val_loss = F.nll_loss(out[proxy_data.val_mask], proxy_data.y[proxy_data.val_mask]).item()
        val_pred = out[proxy_data.val_mask].argmax(dim=1)
        val_acc = val_pred.eq(proxy_data.y[proxy_data.val_mask]).sum().item() / proxy_data.val_mask.sum().item()

    return val_loss, val_acc


def combined_loss_function(proxy_output, original_probs, target_labels, train_mask, a, b):
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    proxy_output_masked = proxy_output[train_mask]  
    original_probs_masked = original_probs[train_mask]  
    target_labels_masked = target_labels[train_mask]  


    kl_div_loss = kl_loss(proxy_output_masked, original_probs_masked)
    ce_loss = cross_entropy_loss(proxy_output_masked, target_labels_masked)

    return a * kl_div_loss + b * ce_loss

def train_proxy_model_on_combined_subgraph(data, combined_subgraph, original_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    combined_subgraph = combined_subgraph.to(device)

    num_classes = combined_subgraph.num_classes if hasattr(combined_subgraph, 'num_classes') else combined_subgraph.y.max().item() + 1

    num_node_features = combined_subgraph.x.shape[1]
    proxy_model = ProxyGNNModel(
        num_node_features=num_node_features,
        hidden_dim=16,
        num_classes=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(proxy_model.parameters(), lr=0.001, weight_decay=1e-4)

    num_nodes = combined_subgraph.num_nodes
    labels = combined_subgraph.y.cpu().numpy()
    

    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    min_count = counts.min()
    
    
    if num_nodes < 10:
        train_indices = np.arange(num_nodes)
        test_indices = np.array([], dtype=int)
        
    elif min_count == 1:
        rare_labels = [label for label, count in label_counts.items() if count == 1]
        rare_indices = np.where(np.isin(labels, rare_labels))[0]
        normal_indices = np.where(~np.isin(labels, rare_labels))[0]
        
        if len(normal_indices) >= 2:
            normal_labels = labels[normal_indices]
            train_normal, test_normal = train_test_split(
                normal_indices,
                test_size=0.2,
                stratify=normal_labels,
                random_state=42
            )
            train_indices = np.concatenate([train_normal, rare_indices])
            test_indices = test_normal
        else:
            train_indices = np.arange(num_nodes)
            test_indices = np.array([], dtype=int)
            
    elif min_count < 5:
        test_size = max(0.1, 1.0 / min_count)
        test_size = min(test_size, 0.3)
        train_indices, test_indices = train_test_split(
            np.arange(num_nodes),
            test_size=test_size,
            stratify=labels,
            random_state=42
        )
    else:
        train_indices, test_indices = train_test_split(
            np.arange(num_nodes),
            test_size=0.2,
            stratify=labels,
            random_state=42
        )
    
    subgraph_n_id = combined_subgraph.n_id
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if len(train_indices) > 0:
        train_mask[train_indices] = True

    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if len(test_indices) > 0:
        test_mask[test_indices] = True

    with torch.no_grad():
        original_logits = original_model(data)
        original_probs = F.softmax(original_logits, dim=1)
    original_probs_subgraph = original_probs[subgraph_n_id]

    proxy_model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        out = proxy_model(combined_subgraph)
        loss = combined_loss_function(out, original_probs_subgraph, combined_subgraph.y, train_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proxy_model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 50 == 0:
            print(f' {epoch + 1},LOSS: {loss.item():.4f}')

    return proxy_model, combined_subgraph