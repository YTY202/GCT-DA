import torch
import numpy as np
from torch_geometric.data import Data

def getData(features_list, edge_index_list, edge_attr):
    x = torch.tensor(features_list, dtype=torch.float)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1

    return link_labels