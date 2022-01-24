# write tests for bfs
import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """ Helper function to check the correctness of the adjacency matrix encoding an MST.
        Note that because the MST of a graph is not guaranteed to be unique, we cannot 
        simply check for equality against a known MST of a graph. 

        Arguments:
            adj_mat: Adjacency matrix of full graph
            mst: Adjacency matrix of proposed minimum spanning tree
            expected_weight: weight of the minimum spanning tree of the full graph
            allowed_error: Allowed difference between proposed MST weight and `expected_weight`

        TODO: 
            Add additional assertions to ensure the correctness of your MST implementation
        For example, how many edges should a minimum spanning tree have? Are minimum spanning trees
        always connected? What else can you think of?
    """
    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    assert np.allclose(mst, mst.T), "Proposed MST is not symmetric"

    assert np.sum(adj_mat) >= np.sum(mst), "Proposed MST has more weight than original graph"


def test_mst_small():
    """ Unit test for the construction of a minimum spanning tree on a small graph """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """ Unit test for the construction of a minimum spanning tree using 
    single cell data, taken from the Slingshot R package 
    (https://bioconductor.org/packages/release/bioc/html/slingshot.html)
    """
    file_path = './data/slingshot_example.txt'
    # load coordinates of single cells in low-dimensional subspace
    coords = np.loadtxt(file_path)
    # compute pairwise distances for all 140 cells to form an undirected weighted graph
    dist_mat = pairwise_distances(coords)
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """ TODO: Write at least one unit test for MST construction """
    # large adj mat
    adj_mat = [[0, 40, 14, 52, 38, 79, 53, 66, 72, 55],
               [40,  0, 44, 25, 81, 34, 54, 59, 56, 65],
               [14, 44, 0, 64, 79, 50, 73, 71, 55, 44],
               [52, 25, 64,  0, 43, 75, 49, 20, 65, 48],
               [38, 81, 79, 43, 0, 71, 48, 38, 40, 33],
               [79, 34, 50, 75, 71, 0, 13, 64, 34, 47],
               [53, 54, 73, 49, 48, 13, 0, 81,  9, 73],
               [66, 59, 71, 20, 38, 64, 81, 0, 19, 55],
               [72, 56, 55, 65, 40, 34,  9, 19, 0, 28],
               [55, 65, 44, 48, 33, 47, 73, 55, 28, 0]]

    g = Graph(np.array(adj_mat))
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 199)

    # small adj mat
    adj_mat = [[1, 1, 3, 4, 4],
               [1, 0, 2, 3, 1],
               [3, 2, 2, 2, 1],
               [4, 3, 2, 3, 2],
               [4, 1, 1, 2, 1]]
    g = Graph(np.array(adj_mat))
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 5)