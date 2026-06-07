# GIST: Towards Memory-Efficient and Computationally Scalable Graph Neural Network Explainability

> A scalable and model-agnostic framework for explaining Graph Neural Networks on large-scale graphs through graph partitioning, random-walk sampling, and surrogate model distillation.

---

## Overview

Graph Neural Network (GNN) explainers such as **GNNExplainer**, **PGExplainer**, **PGMExplainer**, and **GNNShap** have demonstrated strong explanation capability on small and medium-sized graphs. However, their computational and memory requirements grow rapidly with graph size, making them difficult to deploy on real-world graphs containing hundreds of thousands or even millions of nodes.

**GIST** addresses this challenge by constructing a compact proxy graph and training a lightweight surrogate GNN that faithfully reproduces the behavior of the original model. Existing post-hoc explainers can then be applied directly to the surrogate model, significantly reducing memory consumption and runtime while maintaining explanation fidelity.

---

## Key Ideas

GIST consists of three core components:

### 1. Graph Partitioning

The original graph is partitioned into multiple compact subgraphs using **METIS**.

To automatically determine the optimal number of partitions, GIST introduces **Bayesian Optimization** that minimizes the prediction discrepancy between the surrogate model and the original GNN.

### 2. Random-Walk Graph Sampling

Graph partitioning may lose important long-range dependencies.

To preserve global topology information, GIST performs random-walk-based sampling on the original graph and extracts structurally important cross-partition connections.

### 3. Surrogate GNN Distillation

A surrogate GNN is trained on the constructed proxy graph using a hybrid objective:

```math
L_{total}=\alpha L_{KL}+\beta L_{CE}
```

where:

- **Cross-Entropy Loss** preserves node classification performance.
- **KL Divergence Loss** aligns the output distribution of the surrogate model with the original GNN.

This enables faithful explanation generation while operating on substantially smaller graphs.

---

## Framework

```text
Original Graph
       │
       ▼
 ┌─────────────┐
 │   METIS     │
 │ Partition   │
 └─────────────┘
       │
       ▼
 Partitioned Subgraph
       │
       ├─────────────┐
       │             │
       ▼             ▼
 Bayesian      Random Walk
Optimization    Sampling
       │             │
       └──────┬──────┘
              ▼
        Proxy Graph
              │
              ▼
       Surrogate GNN
              │
              ▼
    Existing Explainers
(GNNExplainer / PGExplainer /
 GNNShap / PGMExplainer)
              │
              ▼
      Explanation Results
```

---

## Features

- 🚀 Plug-and-play framework
- 🔌 Compatible with existing post-hoc GNN explainers
- 📈 Supports million-scale graphs
- 🎯 Bayesian partition selection
- 🌐 Random-walk topology preservation
- 🧠 Knowledge-distillation-based surrogate training
- 💾 Significant reduction in GPU memory consumption
- ⚡ Significant explanation speedup

---

## Supported Explainers

| Explainer | Supported |
|------------|------------|
| GNNExplainer | ✅ |
| PGExplainer | ✅ |
| GNNShap | ✅ |
| PGMExplainer | ✅ |
| Curvature-enhanced GNNExplainer | ✅ |

---

## Experimental Results

GIST achieves:

| Metric | Improvement |
|----------|------------|
| Runtime | 1.3× – 2.5× Faster |
| GPU Memory | 40% – 55% Lower |
| Scalability | Up to 2.45M Nodes |
| Fidelity | Comparable or Better |

On large-scale datasets such as **Reddit** and **ogbn-products**, all baseline explainers encounter out-of-memory (OOM) errors, while GIST successfully generates explanations on a single consumer GPU.

---

## Methodology

### Graph Partitioning

GIST first partitions the original graph using METIS.

The partition number is automatically selected using Bayesian Optimization:

```math
k^*=\arg\min_k |Acc_{surrogate}(k)-Acc_{original}|
```

### Graph Sampling

To preserve global topology information, GIST performs random walks on the original graph and extracts structurally important edges.

### Proxy Graph Construction

The final proxy graph is constructed as:

```math
G_{proxy}=G_{partition}\cup G_{sample}
```

### Surrogate Training

The surrogate GNN is optimized using:

```math
L_{total}=\alpha L_{KL}+\beta L_{CE}
```

where:

- \(L_{KL}\) aligns prediction distributions.
- \(L_{CE}\) preserves classification accuracy.

---

## Citation

If you find this project useful, please cite:

```bibtex
@article{gist2026,
  title={Towards Memory-Efficient and Computationally Scalable Graph Neural Network Explainability},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
```

---

## Acknowledgements

This repository builds upon the following excellent works:

- GNNExplainer
- PGExplainer
- GNNShap
- PGMExplainer
- PyTorch Geometric
- METIS

---

## License

This project is released under the MIT License.

```text
MIT License
Copyright (c) 2026
```
