"""
graph_builder.py

Crystal graph construction utilities for PyTorch Geometric.
"""

import torch
from torch_geometric.data import Data
from jarvis.core.atoms import Atoms


def structure_to_graph(atoms_dict, target_value, cutoff: float = 5.0):
    """
    Convert a JARVIS atoms dictionary into a PyTorch Geometric graph.

    Nodes are atoms represented by atomic number.
    Edges connect neighboring atoms within a cutoff radius.
    Edge attributes store interatomic distances.
    """

    atoms_obj = Atoms.from_dict(atoms_dict)
    structure = atoms_obj.pymatgen_converter()

    atomic_numbers = [site.specie.Z for site in structure]
    z = torch.tensor(atomic_numbers, dtype=torch.long)

    edge_index = []
    edge_attr = []

    for i, site_i in enumerate(structure):
        neighbors = structure.get_neighbors(site_i, cutoff)

        for neighbor in neighbors:
            j = neighbor.index
            distance = neighbor.nn_distance

            if i != j:
                edge_index.append([i, j])
                edge_attr.append([distance])

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor([target_value], dtype=torch.float)

    graph = Data(
        z=z,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=len(z)
    )

    return graph


def build_graph_dataset(df, target_column: str, cutoff: float = 5.0, max_samples=None):
    """
    Build a list of PyTorch Geometric graphs from a cleaned materials DataFrame.
    """

    graphs = []

    if max_samples is not None:
        df = df.head(max_samples)

    for _, row in df.iterrows():
        graph = structure_to_graph(
            atoms_dict=row["atoms"],
            target_value=row[target_column],
            cutoff=cutoff
        )

        if graph is not None:
            graphs.append(graph)

    print(f"Total graphs created: {len(graphs)}")

    return graphs
