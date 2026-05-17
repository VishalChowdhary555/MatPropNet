"""
train_gnn.py

Training utilities for the improved crystal GNN.
"""

import torch
import torch.nn as nn
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from src.models.improved_crystal_gnn import ImprovedCrystalGNN
from src.evaluation.metrics import regression_metrics, print_metrics


def normalize_graph_targets(graph_dataset):
    """
    Normalize graph target values using StandardScaler.
    """

    target_scaler = StandardScaler()

    target_values = np.array(
        [g.y.item() for g in graph_dataset]
    ).reshape(-1, 1)

    target_scaler.fit(target_values)

    for graph in graph_dataset:
        normalized_y = target_scaler.transform(
            [[graph.y.item()]]
        )[0][0]

        graph.y = torch.tensor([normalized_y], dtype=torch.float)

    return graph_dataset, target_scaler


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Train GNN for one epoch.
    """

    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(batch)
        loss = loss_fn(pred, batch.y.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def evaluate_gnn_inverse(model, loader, device, target_scaler):
    """
    Evaluate GNN and inverse-transform normalized predictions.
    """

    model.eval()

    y_true_norm = []
    y_pred_norm = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            pred = model(batch)

            y_true_norm.extend(batch.y.view(-1).cpu().numpy())
            y_pred_norm.extend(pred.cpu().numpy())

    y_true_norm = np.array(y_true_norm).reshape(-1, 1)
    y_pred_norm = np.array(y_pred_norm).reshape(-1, 1)

    y_true = target_scaler.inverse_transform(y_true_norm).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_norm).flatten()

    metrics = regression_metrics(y_true, y_pred)

    return metrics, y_true, y_pred


def train_gnn(
    graph_dataset,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
    num_epochs: int = 100,
    patience: int = 15,
    save_path: str = "best_crystal_gnn.pth"
):
    """
    Train improved crystal GNN with early stopping.
    """

    graph_dataset, target_scaler = normalize_graph_targets(graph_dataset)

    train_graphs, test_graphs = train_test_split(
        graph_dataset,
        test_size=test_size,
        random_state=random_state
    )

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImprovedCrystalGNN().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-4
    )

    loss_fn = nn.MSELoss()

    best_mae = float("inf")
    best_epoch = 0
    counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "test_mae": [],
        "test_rmse": [],
        "test_r2": []
    }

    print("Using device:", device)

    for epoch in range(1, num_epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device
        )

        metrics, y_true, y_pred = evaluate_gnn_inverse(
            model,
            test_loader,
            device,
            target_scaler
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_mae"].append(metrics["MAE"])
        history["test_rmse"].append(metrics["RMSE"])
        history["test_r2"].append(metrics["R2"])

        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            best_epoch = epoch
            counter = 0

            torch.save(model.state_dict(), save_path)

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"MAE: {metrics['MAE']:.4f} | "
                f"RMSE: {metrics['RMSE']:.4f} | "
                f"R²: {metrics['R2']:.4f} | Best model saved"
            )

        else:
            counter += 1

            if epoch % 5 == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"MAE: {metrics['MAE']:.4f} | "
                    f"RMSE: {metrics['RMSE']:.4f} | "
                    f"R²: {metrics['R2']:.4f} | "
                    f"Patience: {counter}/{patience}"
                )

        if counter >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(save_path, map_location=device))

    final_metrics, y_true, y_pred = evaluate_gnn_inverse(
        model,
        test_loader,
        device,
        target_scaler
    )

    print_metrics(final_metrics, title="Best Improved Crystal GNN Results")

    return model, target_scaler, final_metrics, history, y_true, y_pred
