# ProxyGNNE

ProxyGNNE is a small Python research project for training and explaining Graph Convolutional Networks on Planetoid graph datasets. The workflow trains an original GCN, generates node explanations with GNNExplainer, builds proxy subgraphs through graph partitioning and GraphSAINT sampling, then trains and evaluates a proxy GNN.

## Project structure

- `main.ipynb` - notebook workflow for running the full experiment.
- `config.yaml` - dataset, model, training, and sampling configuration.
- `models/gnn_models.py` - original GCN, proxy GCN, and explainer wrapper models.
- `utils/` - training, evaluation, partitioning, sampling, graph processing, and explanation utilities.

## Setup

Install the main dependencies:

```bash
pip install torch torch-geometric pymetis numpy matplotlib scikit-learn networkx pyyaml tqdm optuna
```

## Usage

Run the notebook interactively:

```bash
jupyter notebook main.ipynb
```

The default configuration uses Planetoid datasets downloaded under `./dataset`. Edit `config.yaml` to change the dataset, model hidden size, training epochs, optimizer settings, or graph partition search range.
