import numpy as np
import heapq
from typing import Union

class Graph:
    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """ Unlike project 2, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or the path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """ Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. 
        Note that because we assume our input graph is undirected, `self.adj_mat` is symmetric. 
        Row i and column j represents the edge weight between vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        TODO: 
            This function does not return anything. Instead, store the adjacency matrix 
        representation of the minimum spanning tree of `self.adj_mat` in `self.mst`.
        We highly encourage the use of priority queues in your implementation. See the heapq
        module, particularly the `heapify`, `heappop`, and `heappush` functions.
        """

        self.mst = np.zeros(self.adj_mat.shape) # initialize to zero matrix
        num_nodes = self.adj_mat.shape[0]

        # this representation allows us to keep track of a node's weight in addition to the vertices it connects --
        # useful for implementation, note that I'm only storing the nonzero edges here
        edges_with_weights = [[(self.adj_mat[x,y], (x,y)) for y in range(num_nodes) if self.adj_mat[x,y] != 0]
                              for x in range(num_nodes)]

        visited = {0} # represent each node as an index of the matrix
        # outgoing nonzero edges of 0th node stored in the form
        # [(weight1 != 0, (origin1, dest1)), (weight2 != 0, (origin1, dest2)), ...], [...], ...]
        outgoing = edges_with_weights[0] # neighbors of 0th node
        heapq.heapify(outgoing) # in place pq

        while len(visited) < num_nodes:
            lowest_edge_weight = heapq.heappop(outgoing) # this is of the form (weight, (origin, destination))
            destination_node = lowest_edge_weight[1][1]

            if destination_node not in visited:
                self.mst[lowest_edge_weight[1][0], destination_node] = lowest_edge_weight[0] # pull this weight from self.adj_mat
                self.mst[destination_node, lowest_edge_weight[1][0]] = lowest_edge_weight[0] # symmetrical matrix
                visited.add(destination_node)

                for node in edges_with_weights[destination_node]:
                    heapq.heappush(outgoing, node)





