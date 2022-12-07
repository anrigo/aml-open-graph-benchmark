from typing import Union
import networkx as nx
import torch_geometric
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt


def visualize(G: Union[torch_geometric.data.Data, nx.Graph]):
    if isinstance(G, torch_geometric.data.Data):
        G = to_networkx(G, to_undirected=True)

    nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels=False)
    plt.show()
