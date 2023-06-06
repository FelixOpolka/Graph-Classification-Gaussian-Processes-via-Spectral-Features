import numpy as np
import pygsp as gsp
import torch
import torch_geometric as ptg


class SyntheticDataset(ptg.data.InMemoryDataset):
    def __init__(self, data, slices):
        super(SyntheticDataset, self).__init__()
        self.data = data
        self.slices = slices


def sample_combined_graph(N_er=95, N_add=5, p=0.3, fully_connected_add=False, seed=0):
    er_graph = gsp.graphs.ErdosRenyi(N=N_er, p=p, seed=seed)
    er_adj = er_graph.A.toarray()
    add_graph = gsp.graphs.FullConnected(N=N_add) if fully_connected_add is True else gsp.graphs.Ring(N=N_add)
    add_adj = add_graph.A.toarray()

    combined_adj = np.concatenate([er_adj, np.zeros([er_adj.shape[0], add_adj.shape[0]])], axis=1)
    temp_adj = np.concatenate([np.zeros([add_adj.shape[0], er_adj.shape[0]]), add_adj], axis=1)
    combined_adj = np.concatenate([combined_adj, temp_adj], axis=0)

    connecting_node_idx = np.random.RandomState(seed=seed).randint(low=0, high=N_er)
    combined_adj[connecting_node_idx, N_er] = 1.0
    combined_adj[N_er, connecting_node_idx] = 1.0

    combined_graph = gsp.graphs.Graph(combined_adj)
    return combined_graph


def create_synthetic_dataset(generate_sample_f):
    edge_indices, ys, xs, num_nodes = [], [], [], []
    edge_slices, y_slices, x_slices = [0], [0], [0]
    rstate = np.random.RandomState(seed=0)
    for idx in range(200):
        adj, label, node_feats = generate_sample_f(idx, rstate)
        # Graph structure
        edge_index = np.stack(np.where(adj > 0.0), axis=0)
        edge_indices.append(edge_index)
        edge_slices.append(edge_slices[-1] + edge_index.shape[1])
        # Graph label
        ys.append(label)
        # y.append(np.random.randint(low=0, high=2))
        y_slices.append(y_slices[-1] + 1)
        # Node attributes
        if node_feats is not None:
            xs.append(node_feats.reshape([-1, 1]))
            x_slices.append(x_slices[-1] + node_feats.shape[0])
        # Num nodes
        num_nodes.append(adj.shape[0])
    edge_index = torch.from_numpy(np.concatenate(edge_indices, axis=1))
    y = torch.from_numpy(np.array(ys))
    num_nodes = torch.from_numpy(np.array(num_nodes))
    slices = {
        "edge_index": torch.from_numpy(np.array(edge_slices)),
        "y": torch.from_numpy(np.array(y_slices)),
        "num_nodes": torch.from_numpy(np.array(y_slices))
    }
    x = None
    if len(xs) > 0:
        x = torch.from_numpy(np.concatenate(xs, axis=0))
        slices["x"] = torch.from_numpy(np.array(x_slices))
    data = ptg.data.Data(edge_index=edge_index, y=y, x=x, num_nodes=num_nodes)
    ds = SyntheticDataset(data=data, slices=slices)
    return ds


def _ring_vs_clique_sampler(idx, rstate):
    fully_connected_add = True if idx % 2 == 0 else False
    N_er = rstate.randint(low=10, high=30, size=1)[0]
    N_add = rstate.randint(low=5, high=10, size=1)[0]
    combined_graph = sample_combined_graph(N_er=N_er, N_add=N_add, p=0.3, fully_connected_add=fully_connected_add, seed=idx)
    combined_adj = combined_graph.A.toarray()
    label = int(fully_connected_add)
    # signal = combined_adj.sum(axis=1).reshape([-1, 1])
    # signal = (signal - np.mean(signal)) / np.std(signal)
    return combined_adj, label, None


def _sbm_sampler(idx, rstate):
    num_components = 2 if idx % 2 == 0 else 3
    label = idx % 2
    num_nodes = rstate.randint(low=10, high=30, size=1)[0]
    graph = gsp.graphs.StochasticBlockModel(N=num_nodes, k=num_components, p=0.8, q=0.1)
    adj = graph.A.toarray()
    return adj, label, None


def get_synthetic_dataset(dataset_name):
    if dataset_name == "ring_clique":
        return create_synthetic_dataset(_ring_vs_clique_sampler)
    elif dataset_name == "sbm":
        return create_synthetic_dataset(_sbm_sampler)
    else:
        raise NotImplementedError(f"No synthetic data set named {dataset_name}.")


if __name__ == '__main__':
    get_synthetic_dataset("sbm")