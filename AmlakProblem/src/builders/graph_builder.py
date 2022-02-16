from __future__ import annotations
import math
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from src.helpers.config_reader import ConfigReader

from src.helpers.data_helper import DataHelper
from src.models.normalize_types import NormalizeTypes
from src.models.distance_types import DistanceTypes
from src.helpers.distance_calculator import DistanceCalculator
from src.algorithms.kruskal import Kruskal
from tqdm import tqdm


class GraphBuilder:
    def __init__(
        self, 
        data: pd.DataFrame,
        distance_type: str=DistanceTypes.EUCLIDIAN
    ) -> None:
        self.graphs = []
        self.data = data
        self.distance_type = distance_type

    def normalize_data(self) -> GraphBuilder:
        self.data = DataHelper.normalize(data=self.data, mode=NormalizeTypes.MAX_MIN)
        self.data = self.data.fillna(0).to_numpy()
        return self
    
    def cluster_data(self) -> GraphBuilder:
        print('in the clustering')
        graph = nx.Graph()
        tree = KDTree(self.data)
        n = self.data.shape[0]
        for u in tqdm(range(n)):
            dist, adj = tree.query(self.data[u, :].reshape(1, -1), k=22)
            dist, adj = dist[0], adj[0]
            for i, v in enumerate(adj):
                if u == v:
                    continue
                graph.add_edge(u, v, weight=dist[i])
        self.clusters = Kruskal(graph).clustering(max_cluster_size=ConfigReader.read('features.similarity.cluster_count'))
        return self
    
    def build_graphs(self) -> GraphBuilder:
        print('in the building graph')
        self.graphs = []
        for cluster in self.clusters:
            graph = nx.Graph()
            for u in cluster:
                graph.add_node(u)
            self.graphs.append(graph)
        return self

    def build_adjacency(self) -> GraphBuilder:
        print('in the building adjacency')
        # creating weighted adj
        for graph in tqdm(self.graphs):
            for i, u in enumerate(graph.nodes):
                for j, v in enumerate(graph.nodes):
                    if i >= j:
                        continue
                    # print('(%2d -> %2d, %.02f)' % (u, v, DistanceCalculator.euclidian(self.data[u, :], self.data[v, :])))
                    graph.add_edge(u, v, weight=DistanceCalculator.calculate(
                        self.data[u, :], self.data[v, :], distance_type=self.distance_type)
                    )
        return self

    def normalize_weights(self) -> GraphBuilder:
        for graph in self.graphs:
            weights = pd.DataFrame([data_edge['weight'] for u, v, data_edge in graph.edges(data=True)], columns=['weights'])
            weights = DataHelper.normalize(data=weights, mode=NormalizeTypes.BOTH)
            weights = weights.to_numpy()
            attributes = {}
            for index, (u, v) in enumerate(graph.edges):
                attributes[(u, v)] = weights[index, 0]
            nx.set_edge_attributes(graph, attributes, 'weight')

        return self
    
    def print_edges(self) -> GraphBuilder:
        print()
        print()
        print()
        print()
        print()
        print('PRINTING EDGES')
        for graph in self.graphs:
            for u, v, data_edge in graph.edges(data=True):
                print('%2d->%2d, %.03f' % (u, v, data_edge['weight']))
        return self

    def get_degree_centrality(self, threshold: float) -> pd.DataFrame:
        print('in the degree centrality')
        data = np.zeros((len(self.data), 1))
        for graph in tqdm(self.graphs):
            graph_prim = nx.Graph()
            for u, v, data_edge in graph.edges(data=True):
                if data_edge['weight'] < threshold:
                    graph_prim.add_edge(u, v, weight=data_edge['weight'])
            temp = nx.degree_centrality(graph_prim)
            for u in temp.keys():
                data[u, 0] = temp[u]
        return pd.DataFrame(data, columns=[f'{self.distance_type.value.lower()}_degree_centrality'])
    
    def get_mean_weights(self) -> pd.DataFrame:
        print('in the mean weights')
        data = np.zeros((len(self.data), 1))
        for graph in tqdm(self.graphs):
            for u, v, data_edge in graph.edges(data=True):
                data[u, 0] += data_edge['weight']
                data[v, 0] += data_edge['weight']
            for u in graph.nodes:
                data[u, 0] /= graph.number_of_nodes()
        return pd.DataFrame(data, columns=[f'{self.distance_type.value.lower()}_mean_weights'])

    def get_closeness_centrality(self, threshold: float) -> pd.DataFrame:
        print('in the closeness centrality')
        data = np.zeros((len(self.data), 1))
        for graph in tqdm(self.graphs):
            graph_prim = nx.Graph()
            for u, v, data_edge in graph.edges(data=True):
                if data_edge['weight'] < threshold:
                    graph_prim.add_edge(u, v, weight=data_edge['weight'])
            temp = nx.closeness_centrality(graph_prim, distance='weight')
            for u in temp.keys():
                data[u, 0] = temp[u]
        return pd.DataFrame(data, columns=[f'{self.distance_type.value.lower()}_closeness_centrality'])

    def get_betweenness_centrality(self, threshold: float) -> pd.DataFrame:
        print('in the betweenness centrality')
        data = np.zeros((len(self.data), 1))
        for graph in tqdm(self.graphs):
            graph_prim = nx.Graph()
            for u, v, data_edge in graph.edges(data=True):
                if data_edge['weight'] < threshold:
                    graph_prim.add_edge(u, v, weight=data_edge['weight'])
            temp = nx.betweenness_centrality(graph_prim, weight='weight')
            for u in temp.keys():
                data[u, 0] = temp[u]
        return pd.DataFrame(data, columns=[f'{self.distance_type.value.lower()}_betweenness_centrality'])
     
    def get_harmonic_centrality(self, threshold: float) -> pd.DataFrame:
        print('in the harmonic centrality')
        data = np.zeros((len(self.data), 1))
        for graph in tqdm(self.graphs):
            graph_prim = nx.Graph()
            for u, v, data_edge in graph.edges(data=True):
                if data_edge['weight'] < threshold:
                    graph_prim.add_edge(u, v, weight=data_edge['weight'])
            temp = nx.harmonic_centrality(graph_prim, distance='weight')
            for u in temp.keys():
                data[u, 0] = temp[u]
        return pd.DataFrame(data, columns=[f'{self.distance_type.value.lower()}_harmonic_centrality'])
     
    def get_load_centrality(self, threshold: float) -> pd.DataFrame:
        print('in the load centrality')
        data = np.zeros((len(self.data), 1))
        for graph in tqdm(self.graphs):
            graph_prim = nx.Graph()
            for u, v, data_edge in graph.edges(data=True):
                if data_edge['weight'] < threshold:
                    graph_prim.add_edge(u, v, weight=data_edge['weight'])
            temp = nx.load_centrality(graph_prim, weight='weight')
            for u in temp.keys():
                data[u, 0] = temp[u]
        return pd.DataFrame(data, columns=[f'{self.distance_type.value.lower()}_load_centrality'])

    def get_subgraph_centrality(self, threshold: float) -> pd.DataFrame:
        print('in the subgraph centrality')
        data = np.zeros((len(self.data), 1))
        for graph in tqdm(self.graphs):
            graph_prim = nx.Graph()
            for u, v, data_edge in graph.edges(data=True):
                if data_edge['weight'] < threshold:
                    graph_prim.add_edge(u, v, weight=data_edge['weight'])
            temp = nx.subgraph_centrality(graph_prim)
            for u in temp.keys():
                data[u, 0] = temp[u]
        return pd.DataFrame(data, columns=[f'{self.distance_type.value.lower()}_subgraph_centrality'])
    
    def get_second_order_centrality(self, threshold: float) -> pd.DataFrame:
        print('in the second order centrality')
        data = np.zeros((len(self.data), 1))
        for graph in tqdm(self.graphs):
            graph_prim = nx.Graph()
            for u, v, data_edge in graph.edges(data=True):
                if data_edge['weight'] < threshold:
                    graph_prim.add_edge(u, v, weight=data_edge['weight'])
            temp = nx.second_order_centrality(graph_prim)
            for u in temp.keys():
                data[u, 0] = temp[u]
        return pd.DataFrame(data, columns=[f'{self.distance_type.value.lower()}_second_order_centrality'])

