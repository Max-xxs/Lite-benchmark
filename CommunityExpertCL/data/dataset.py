"""
Dataset loader for CommunityExpertCL.
Supports: cora, citeseer, cora-full, coauthor-cs, amazon-computers, wikics,
          reddit, ogbn-arxiv, ogbn-products (all via PyG, all undirected).
"""

import heapq
import os
import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch.utils.data import Dataset

try:
    from ogb.nodeproppred import PygNodePropPredDataset
    OGB_AVAILABLE = True
except ImportError:
    OGB_AVAILABLE = False

try:
    import dgl
    from dgl.data import (
        AmazonCoBuyComputerDataset,
        CoauthorCSDataset,
        CoraFullDataset,
        RedditDataset,
    )
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False

try:
    from ogb.nodeproppred import DglNodePropPredDataset
    OGB_DGL_AVAILABLE = True
except ImportError:
    OGB_DGL_AVAILABLE = False


DATASET_ALIASES = {
    'Arxiv-CL': 'ogbn-arxiv',
    'Products-CL': 'ogbn-products',
    'Reddit-CL': 'reddit',
    'CoraFull-CL': 'cora-full',
    'Corafull-CL': 'cora-full',
    'CoauthorCS-CL': 'coauthor-cs',
    'AmazonComputers-CL': 'amazon-computers',
}

CGLB_PROTOCOL_DATASETS = {
    'cora-full',
    'coauthor-cs',
    'amazon-computers',
    'reddit',
    'ogbn-arxiv',
    'ogbn-products',
}

SUPPORTED_DATASETS = {
    'cora', 'citeseer', 'cora-full', 'coauthor-cs', 'amazon-computers',
    'wikics', 'reddit', 'ogbn-arxiv', 'ogbn-products',
}


class GraphDataset(Dataset):
    """Undirected graph dataset for expert-based continual learning."""

    def __init__(self, dataset, data_path, svd_dim=0,
                 reorder_by_class_size=False,
                 data_protocol='native'):
        self.dataset = DATASET_ALIASES.get(dataset, dataset)
        self.requested_dataset = dataset
        self.data_path = data_path
        self.svd_dim = svd_dim
        self.reorder_by_class_size = reorder_by_class_size
        self.data_protocol = data_protocol
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Supported: {SUPPORTED_DATASETS}"
            )
        if data_protocol not in {'native', 'cglb'}:
            raise ValueError(
                f"Unknown data_protocol '{data_protocol}'. "
                "Use 'native' or 'cglb'."
            )
        self.data, self.id_by_class = self._load_data()

    def __getitem__(self, idx):
        return {
            'node_id': idx,
            'labels': self.data.y[idx].to(torch.long),
        }

    def __len__(self):
        return self.data.x.shape[0]

    def _load_data(self):
        if (self.data_protocol == 'cglb'
                and self.dataset in CGLB_PROTOCOL_DATASETS):
            return self._load_cglb_protocol()

        if self.dataset in ('cora', 'citeseer'):
            return self._load_planetoid()
        elif self.dataset == 'cora-full':
            return self._load_cora_full()
        elif self.dataset == 'coauthor-cs':
            return self._load_coauthor()
        elif self.dataset == 'amazon-computers':
            return self._load_amazon()
        elif self.dataset == 'wikics':
            return self._load_wikics()
        elif self.dataset == 'reddit':
            return self._load_reddit()
        elif self.dataset in ('ogbn-arxiv', 'ogbn-products'):
            return self._load_ogbn()

    @staticmethod
    def _instantiate_with_optional_raw_dir(dataset_cls, raw_dir, **kwargs):
        try:
            return dataset_cls(raw_dir=raw_dir, **kwargs)
        except TypeError:
            return dataset_cls(**kwargs)

    @staticmethod
    def _label_from_dgl_graph(graph, fallback_label=None):
        if fallback_label is not None:
            label = fallback_label
        elif 'label' in graph.ndata:
            label = graph.ndata['label']
        elif hasattr(graph, 'dstdata') and 'label' in graph.dstdata:
            label = graph.dstdata['label']
        else:
            raise KeyError("Could not find labels in DGL graph data")
        return label.squeeze(-1).to(torch.long)

    def _dgl_to_pyg_data(self, graph, label=None):
        src, dst = graph.edges()
        edge_index = torch.stack([src, dst], dim=0)
        x = graph.ndata['feat']
        y = self._label_from_dgl_graph(graph, fallback_label=label)
        return Data(x=x, edge_index=edge_index, y=y)

    def _load_cglb_protocol(self):
        """
        Load datasets with the same source family used by CGLB.

        This keeps the original edge orientation and class labels, then adds
        self-loops after task subgraph construction just like the native path.
        """
        if self.dataset in ('ogbn-arxiv', 'ogbn-products'):
            if not OGB_DGL_AVAILABLE:
                raise ImportError(
                    "ogb + dgl are required for data_protocol='cglb' "
                    "on OGB datasets."
                )
            print(f"Loading {self.dataset} with CGLB-compatible DGL OGB loader...")
            ogb_root = os.path.join(self.data_path, 'ogb_downloaded')
            dgl_dataset = DglNodePropPredDataset(
                name=self.dataset,
                root=ogb_root,
            )
            graph, label = dgl_dataset[0]
            return self._process(
                self._dgl_to_pyg_data(graph, label=label),
                make_undirected=False,
                remove_isolated=False,
            )

        if not DGL_AVAILABLE:
            raise ImportError(
                "dgl is required for data_protocol='cglb' on this dataset."
            )

        if self.dataset == 'cora-full':
            print("Loading CoraFull with CGLB-compatible DGL loader...")
            dgl_dataset = self._instantiate_with_optional_raw_dir(
                CoraFullDataset,
                self.data_path,
            )
            graph = dgl_dataset[0]
        elif self.dataset == 'coauthor-cs':
            print("Loading Coauthor CS with CGLB-compatible DGL loader...")
            dgl_dataset = self._instantiate_with_optional_raw_dir(
                CoauthorCSDataset,
                self.data_path,
            )
            graph = dgl_dataset[0]
        elif self.dataset == 'amazon-computers':
            print("Loading Amazon Computers with CGLB-compatible DGL loader...")
            dgl_dataset = self._instantiate_with_optional_raw_dir(
                AmazonCoBuyComputerDataset,
                self.data_path,
            )
            graph = dgl_dataset[0]
        elif self.dataset == 'reddit':
            print("Loading Reddit with CGLB-compatible DGL loader...")
            dgl_dataset = self._instantiate_with_optional_raw_dir(
                RedditDataset,
                self.data_path,
                self_loop=False,
            )
            graph = dgl_dataset.graph
            label = dgl_dataset.labels.view(-1)
            return self._process(
                self._dgl_to_pyg_data(graph, label=label),
                make_undirected=False,
                remove_isolated=False,
            )
        else:
            raise ValueError(
                f"Dataset '{self.dataset}' is not supported by the CGLB protocol"
            )

        return self._process(
            self._dgl_to_pyg_data(graph),
            make_undirected=False,
            remove_isolated=False,
        )

    def _load_planetoid(self):
        from torch_geometric.datasets import Planetoid
        name_map = {'cora': 'Cora', 'citeseer': 'CiteSeer'}
        print(f"Loading {self.dataset} from PyG Planetoid...")
        pyg_dataset = Planetoid(root=self.data_path, name=name_map[self.dataset])
        return self._process(pyg_dataset[0])

    def _load_cora_full(self):
        from torch_geometric.datasets import CitationFull
        print("Loading CoraFull from PyG CitationFull...")
        pyg_dataset = CitationFull(root=self.data_path, name='Cora')
        return self._process(pyg_dataset[0])

    def _load_coauthor(self):
        from torch_geometric.datasets import Coauthor
        print("Loading Coauthor CS from PyG...")
        pyg_dataset = Coauthor(root=self.data_path, name='CS')
        return self._process(pyg_dataset[0])

    def _load_amazon(self):
        from torch_geometric.datasets import Amazon
        print("Loading Amazon Computers from PyG...")
        pyg_dataset = Amazon(root=self.data_path, name='Computers')
        return self._process(pyg_dataset[0])

    def _load_wikics(self):
        from torch_geometric.datasets import WikiCS
        print("Loading WikiCS from PyG...")
        pyg_dataset = WikiCS(root=self.data_path)
        return self._process(pyg_dataset[0])

    def _load_reddit(self):
        from torch_geometric.datasets import Reddit
        print("Loading Reddit from PyG...")
        pyg_dataset = Reddit(root=self.data_path)
        return self._process(pyg_dataset[0])

    def _load_ogbn(self):
        if not OGB_AVAILABLE:
            raise ImportError("ogb not installed. Install with: pip install ogb")
        print(f"Loading OGB dataset: {self.dataset} ...")
        ogb_dataset = PygNodePropPredDataset(
            name=self.dataset, root=self.data_path
        )
        return self._process(ogb_dataset[0])

    def _process(self, data, make_undirected=True, remove_isolated=True):
        """Common processing: optional symmetrization/isolate removal, self-loops, reorder."""
        data.y = data.y.to(torch.long)
        if data.y.dim() > 1:
            data.y = data.y.squeeze(-1)

        edge_index = data.edge_index
        if make_undirected:
            edge_index = to_undirected(edge_index)

        x, y = data.x, data.y
        if remove_isolated:
            edge_index, x, y = self._remove_isolated_nodes(edge_index, x, y)

        self.original_edge_index = edge_index

        if self.svd_dim > 0 and x.size(1) > self.svd_dim:
            print(f"Applying Truncated SVD: {x.size(1)} -> {self.svd_dim} dims...")
            svd = TruncatedSVD(n_components=self.svd_dim, random_state=0)
            x = torch.tensor(svd.fit_transform(x.numpy()), dtype=x.dtype)
            explained = svd.explained_variance_ratio_.sum() * 100
            print(f"  Explained variance: {explained:.1f}%")

        # Keep only the no-self-loop graph globally. TaskLoader will attach
        # self-loops lazily per subgraph to avoid duplicating giant edge tensors
        # for Reddit/Products-scale graphs.
        new_data = Data(x=x, edge_index=edge_index, y=y)

        id_by_class = self._build_class_index(new_data)
        self._print_info(new_data, id_by_class)
        return new_data, id_by_class

    @staticmethod
    def _remove_isolated_nodes(edge_index, x, y):
        num_nodes = x.size(0)
        has_edge = torch.zeros(num_nodes, dtype=torch.bool)
        has_edge[edge_index[0].unique()] = True
        has_edge[edge_index[1].unique()] = True

        num_removed = (~has_edge).sum().item()
        if num_removed > 0:
            print(f"Removing {num_removed} isolated nodes...")
            old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
            old_indices = torch.where(has_edge)[0]
            old_to_new[old_indices] = torch.arange(old_indices.size(0))

            x = x[has_edge]
            y = y[has_edge]

            new_src = old_to_new[edge_index[0]]
            new_dst = old_to_new[edge_index[1]]
            valid = (new_src >= 0) & (new_dst >= 0)
            edge_index = torch.stack([new_src[valid], new_dst[valid]])

        return edge_index, x, y

    def _build_class_index(self, data):
        """
        Build a compact class index.

        By default, labels are compacted while preserving their original order.
        When ``reorder_by_class_size`` is enabled, larger classes are moved to
        smaller indices to reproduce the legacy behavior of this repository.
        """
        labels = data.y
        class_list = sorted(int(i) for i in labels.unique().tolist())
        id_by_class = {int(i): [] for i in class_list}
        for idx, cla in enumerate(labels):
            id_by_class[cla.item()].append(idx)

        if self.reorder_by_class_size:
            ordered_classes = [
                cls for cls, _ in heapq.nlargest(
                    len(class_list),
                    id_by_class.items(),
                    key=lambda item: len(item[1]),
                )
            ]
        else:
            ordered_classes = class_list

        class_mapping = {
            old_id: new_id for new_id, old_id in enumerate(ordered_classes)
        }

        for old_id, new_id in class_mapping.items():
            labels[id_by_class[old_id]] = new_id

        class_list = sorted(int(i) for i in labels.unique().tolist())
        id_by_class = {int(i): [] for i in class_list}
        for idx, cla in enumerate(labels):
            id_by_class[cla.item()].append(idx)

        return id_by_class

    def _print_info(self, data, id_by_class):
        graph_view = 'undirected' if self.data_protocol == 'native' else 'CGLB graph protocol'
        print(f"\n{'='*50}")
        print(f"Dataset: {self.dataset} ({graph_view})")
        print(f"  Data protocol: {self.data_protocol}")
        print(f"  Nodes: {data.x.shape[0]}")
        print(f"  Edges (with self-loops): {data.edge_index.shape[1]}")
        print(f"  Edges (no self-loops): {self.original_edge_index.shape[1]}")
        print(f"  Classes: {data.y.max().item() + 1}")
        print(f"  Feature dim: {data.x.shape[1]}")
        print(f"  Samples per class:")
        for cls in sorted(id_by_class.keys()):
            print(f"    Class {cls}: {len(id_by_class[cls])}")
        print(f"{'='*50}\n")
