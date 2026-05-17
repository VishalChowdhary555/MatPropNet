# MatPropNet  
### Machine Learning and Graph Neural Network Framework for Materials Property Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![PyG](https://img.shields.io/badge/PyTorch%20Geometric-Graph%20Neural%20Networks-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green.svg)
![Materials%20AI](https://img.shields.io/badge/Materials-Informatics-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

# Overview

MatPropNet is a materials informatics framework for predicting material properties using both:

- Classical machine learning
- Crystal graph neural networks (GNNs)

The project combines:

- JARVIS-DFT materials datasets
- Composition-based descriptors
- Graph neural networks for crystal structures
- Explainable AI and feature analysis
- Scientific benchmarking and visualization

The framework predicts important materials properties including:

- Formation energy
- Bandgap
- Bulk modulus
- Shear modulus

---

# Key Features

## Classical Materials Machine Learning
- Composition-based descriptors using `matminer`
- XGBoost baseline models
- Feature scaling and preprocessing
- Feature importance analysis
- Scientific regression benchmarking

## Crystal Graph Neural Networks
- Crystal graph construction from atomic structures
- Atomic embedding layers
- Gaussian radial basis edge expansion
- Residual message passing blocks
- Global graph pooling
- Early stopping and checkpoint saving

## Scientific Evaluation
- MAE / RMSE / R² metrics
- Parity plots
- Residual analysis
- Model comparison studies

---

# Repository Structure

```text
MatPropNet/
│
├── notebooks/
│   └── MatPropNet_revised_clean.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── results/
│   ├── figures/
│   ├── models/
│   └── tables/
│
├── reports/
│
├── app/
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# Dataset

This project uses the:

## JARVIS-DFT Dataset

Source:
https://jarvis.nist.gov/

The dataset contains:
- Crystal structures
- Density functional theory (DFT) calculations
- Electronic properties
- Elastic properties
- Thermodynamic properties

---

# Machine Learning Pipeline

## 1. Data Preprocessing
- Missing value handling
- Physical validity filtering
- Numeric conversion
- Dataset cleaning

## 2. Composition Feature Engineering
Descriptors generated using:
- Stoichiometric descriptors
- Magpie elemental statistics
- Valence orbital descriptors
- Ion-property descriptors

## 3. XGBoost Baseline
Gradient boosted tree regression using engineered descriptors.

## 4. Crystal Graph Construction
Materials converted into graphs:
- Nodes → atoms
- Edges → neighboring atomic interactions
- Edge attributes → interatomic distances

## 5. Improved Crystal GNN
The GNN architecture includes:
- Atomic embeddings
- Gaussian distance expansion
- Residual graph convolutions
- Batch normalization
- Global mean/max pooling

---

# Results

## XGBoost Baseline

| Metric | Value |
|---|---:|
| MAE | 0.1339 |
| RMSE | 0.2292 |
| R² | 0.9530 |

## Improved Crystal GNN

| Metric | Value |
|---|---:|
| MAE | 0.1520 |
| RMSE | 0.2295 |
| R² | 0.9482 |

---

# Example Evaluation Plots

## Improved Crystal GNN Parity Plot

<img width="2100" height="2100" alt="improved_gnn_parity_plot" src="https://github.com/user-attachments/assets/ee56527e-780a-45ab-8198-3440873c3e4e" />



## Residual Distribution

<img width="2400" height="1500" alt="improved_gnn_residual_distribution" src="https://github.com/user-attachments/assets/75500ea3-6974-47f8-aaf8-11074e7f6f63" />


---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/MatPropNet.git
cd MatPropNet
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

## Run XGBoost Pipeline

```python
from src.training.train_xgboost import train_xgboost
```

## Run Crystal GNN Pipeline

```python
from src.training.train_gnn import train_gnn
```

---

# Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- XGBoost
- matminer
- pymatgen
- jarvis-tools
- scikit-learn
- matplotlib

---

# Future Work

Planned future extensions include:

- Multi-property prediction
- Bayesian optimization for material discovery
- SHAP explainability analysis
- Advanced architectures:
  - CGCNN
  - MEGNet
  - ALIGNN
  - Matformer
- Materials recommendation engine
- Streamlit deployment dashboard

---

# Research Significance

This project demonstrates the transition from:

## Classical Materials Informatics
Feature-engineered descriptor models

to:

## Geometric Deep Learning
Crystal graph neural networks capable of learning directly from atomic structures.

---

# Citation

If you use this repository, please cite:

```bibtex
@software{matpropnet,
  title={MatPropNet: Materials Property Prediction using Machine Learning and Graph Neural Networks},
  author={Vishal Chowdhary},
  year={2026}
}
```

---

# License

This project is licensed under the MIT License.
