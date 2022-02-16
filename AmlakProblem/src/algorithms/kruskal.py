import networkx as nx


class Kruskal:
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.pars = [u for u in range(self.n)]
        self.sets_size = [1 for _ in range(self.n)]
    
    def clustering(self, max_cluster_size: int) -> list:
        # edges = self.graph.edges
        # sort edges
        edges = sorted(self.graph.edges(data=True), key=lambda edge: edge[2]['weight'])
        # walk through edges, joining comps
        for u, v, data_edge in edges:
            # print('%2d->%2d, %02f' % (u, v, data_edge['weight']))
            if self.find_par(u) == self.find_par(v):
                continue
            if self.get_size(u) + self.get_size(v) < max_cluster_size:
                # print('JUST JOINED %2d->%2d' % (self.find_par(u), self.find_par(v)))
                self.join(u, v)
        # walk through, find each ones par
        indices = []
        for u in range(self.n):
            indices.append(self.find_par(u))
        # map parents to (0, k)
        cluster_count = 0
        cluster_map = {}
        for cluster in indices:
            if cluster in cluster_map:
                continue
            cluster_map[cluster] = len(cluster_map)
            cluster_count = max(cluster_count, cluster_map[cluster] + 1)
        # building list of nodes for each component
        print('clusters count: ', cluster_count)
        clusters = [[] for _ in range(cluster_count)]
        for index, cluster in enumerate(indices):
            clusters[cluster_map[cluster]].append(index)
        return clusters
    
    def join(self, u: int, v: int) -> None:
        if self.find_par(u) == self.find_par(v):
            return
        if self.get_size(v) < self.get_size(u):
            u, v = v, u
        # size[u] < size[v]
        self.sets_size[self.find_par(v)] += self.sets_size[self.find_par(u)]
        self.pars[self.pars[u]] = self.find_par(v)

    def find_par(self, u: int) -> int:
        if u != self.pars[u]:
            self.pars[u] = self.find_par(self.pars[u])
        return self.pars[u]
    
    def get_size(self, u: int) -> int:
        return self.sets_size[self.find_par(u)]
