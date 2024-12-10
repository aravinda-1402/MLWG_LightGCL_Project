# Graph Contrastive Learning for Optimizing Sparse Data in Recommender Systems with LightGCL

This repository contains the PyTorch implementation of **LightGCL**, a lightweight and efficient graph contrastive learning model designed to address challenges in recommender systems, such as data sparsity and popularity bias. This project reproduces and extends the framework introduced in the paper [**LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation**](https://openreview.net/forum?id=FKXVK9dyMM), presented at the *International Conference on Learning Representations (ICLR)*, 2023.

This implementation was completed as the final project for **CSC 591 (038) Machine Learning with Graphs** at **North Carolina State University** for **Fall 2024**.


## Overview

**LightGCL** leverages **Singular Value Decomposition (SVD)** for robust augmentation, integrating global collaborative signals into the representation learning process without relying on handcrafted augmentations. By addressing key limitations of prior methods, LightGCL achieves superior performance in sparse data scenarios while maintaining computational efficiency.

### Key Features

- **SVD-Based Augmentation**: Captures global collaborative relations for refined user-item interactions.
- **Lightweight Contrastive Learning**: Simplifies the framework while maintaining robustness and scalability.
- **Reproducibility**: The implementation adheres to the original framework with detailed experiments on benchmark datasets.

---

## Directory Structure

```plaintext
MLWG_LightGCL_Project/
│
├── data/              # Scripts and utilities for dataset preprocessing and management
├── log/               # Execution logs, training progress, and validation metrics
├── saved_model/       # Checkpoints of trained models for reuse and evaluation
├── main.py            # Main script orchestrating the pipeline (data loading, training, evaluation)
├── model.py           # LightGCL framework definition (GNN layers, SVD augmentation, loss functions)
├── parser.py          # Command-line argument parser for configuring experiments
├── utils.py           # Utility functions for data manipulation, metrics computation, and visualization
├── README.md          # Documentation with setup instructions and usage guidelines
```

## Running environment

We developed our codes in the following environment:

```
Python version 3.9.12
torch==1.12.0+cu113
numpy==1.21.5
tqdm==4.64.0
```

### 3. How to run the codes

* Yelp
```
python main.py --data yelp
```

* Gowalla

```
python main.py --data gowalla --lambda2 0
```

* ML-10M
```
python main.py --data ml10m --temp 0.5
```

* Tmall

```
python main.py --data tmall --gnn_layer 1
```

* Amazon

```
python main.py --data amazon --gnn_layer 1 --lambda2 0 --temp 0.1
```

### 4. Some configurable arguments

* `--cuda` specifies which GPU to run on if there are more than one.
* `--data` selects the dataset to use.
* `--lambda1` specifies $\lambda_1$, the regularization weight for CL loss.
* `--lambda2` is $\lambda_2$, the L2 regularization weight.
* `--temp` specifies $\tau$, the temperature in CL loss.
* `--dropout` is the edge dropout rate.
* `--q` decides the rank q for SVD.
