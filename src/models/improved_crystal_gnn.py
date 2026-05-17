"""
improved_crystal_gnn.py

Improved crystal graph neural network using atomic embeddings,
Gaussian distance expansion, residual NNConv blocks, and graph pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv, global_mean_pool, global_max_pool


class GaussianDistanceExpansion(nn.Module):
    """
    Expand scalar interatomic distances into Gaussian radial basis features.
    """

    def __init__(self, min_dist=0.0, max_dist=5.0, num_gaussians=50):
        super().__init__()

        centers = torch.linspace(min_dist, max_dist, num_gaussians)

        self.register_buffer("centers", centers)
        self.gamma = 10.0

    def forward(self, distances):
        distances = distances.view(-1, 1)

        return torch.exp(
            -self.gamma * (distances - self.centers) ** 2
        )


class ImprovedCrystalGNN(nn.Module):
    """
    Crystal graph neural network for materials property prediction.
    """

    def __init__(
        self,
        max_atomic_number=100,
        embedding_dim=64,
        hidden_dim=128,
        edge_dim=50
    ):
        super().__init__()

        self.atom_embedding = nn.Embedding(
            max_atomic_number + 1,
            embedding_dim
        )

        self.edge_expansion = GaussianDistanceExpansion(
            min_dist=0.0,
            max_dist=5.0,
            num_gaussians=edge_dim
        )

        self.node_encoder = nn.Linear(embedding_dim, hidden_dim)

        self.edge_network_1 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )

        self.edge_network_2 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )

        self.edge_network_3 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )

        self.conv1 = NNConv(hidden_dim, hidden_dim, self.edge_network_1, aggr="mean")
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.edge_network_2, aggr="mean")
        self.conv3 = NNConv(hidden_dim, hidden_dim, self.edge_network_3, aggr="mean")

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        z = data.z
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x = self.atom_embedding(z)
        x = self.node_encoder(x)
        x = F.relu(x)

        edge_feat = self.edge_expansion(edge_attr)

        h = self.conv1(x, edge_index, edge_feat)
        h = self.bn1(h)
        h = F.relu(h)
        x = x + h

        h = self.conv2(x, edge_index, edge_feat)
        h = self.bn2(h)
        h = F.relu(h)
        x = x + h

        h = self.conv3(x, edge_index, edge_feat)
        h = self.bn3(h)
        h = F.relu(h)
        x = x + h

        pooled_mean = global_mean_pool(x, batch)
        pooled_max = global_max_pool(x, batch)

        graph_embedding = torch.cat(
            [pooled_mean, pooled_max],
            dim=1
        )

        out = self.regressor(graph_embedding)

        return out.view(-1)
