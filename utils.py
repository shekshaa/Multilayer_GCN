import numpy as np
import pandas as pd
import scipy.sparse as sp
from gcn.utils import sparse_to_tuple


def get_indices(df):
    indices = []
    for item in df.values:
        s = item[0].split(' ')
        indices.append([int(s[0]), int(s[1])])
    return np.asarray(indices)


def make_sparse_matrix(indices, shape):
    data = np.ones(indices.shape[0])
    return sp.csr_matrix((data, (indices[:, 0], indices[:, 1])), shape=shape)


def make_node_types(n0, n1, n2):
    n_total = n0 + n1 + n2
    node_types = np.zeros((n_total,), dtype=int)
    node_types[: n0] = 0
    node_types[n0: n0 + n1] = 1
    node_types[n0 + n1:] = 2
    one_hot_node_types = np.zeros((n_total, 3), dtype=int)
    one_hot_node_types[np.arange(n_total), node_types] = 1
    return one_hot_node_types


def make_one_hot(labels):
    max_label = np.max(labels)
    n_nodes = labels.shape[0]
    one_hot_labels = np.zeros((n_nodes, max_label))
    one_hot_labels[np.arange(labels.shape[0]), labels - 1] = 1
    return one_hot_labels


def load_aminer():
    fdf0 = pd.read_csv('./data/aminer/aminer.feat0', delimiter=' ', header=None)
    fdf1 = pd.read_csv('./data/aminer/aminer.feat1', delimiter=' ', header=None)
    fdf2 = pd.read_csv('./data/aminer/aminer.feat2', delimiter=' ', header=None)

    labels = np.concatenate((fdf0.values[:, -1], fdf1.values[:, -1], fdf2.values[:, -1]), axis=0)
    one_hot_labels = make_one_hot(labels)

    df0 = pd.read_csv('./data/aminer/aminer.adj0')
    df1 = pd.read_csv('./data/aminer/aminer.adj1')
    df2 = pd.read_csv('./data/aminer/aminer.adj2')

    adj0_indices = get_indices(df0)
    adj1_indices = get_indices(df1)
    adj2_indices = get_indices(df2)

    n0 = np.max(adj0_indices, axis=0)[0] + 1
    n1 = np.max(adj1_indices, axis=0)[0] + 1
    n2 = np.max(adj2_indices, axis=0)[0] + 1

    adj0 = make_sparse_matrix(adj0_indices, (n0, n0))
    adj1 = make_sparse_matrix(adj1_indices, (n1, n1))
    adj2 = make_sparse_matrix(adj2_indices, (n2, n2))

    df01 = pd.read_csv('./data/aminer/aminer.bet0_1')
    df02 = pd.read_csv('./data/aminer/aminer.bet0_2')

    adj01_indices = get_indices(df01)
    adj02_indices = get_indices(df02)

    adj01 = make_sparse_matrix(adj01_indices, (n0, n1))
    adj02 = make_sparse_matrix(adj02_indices, (n0, n2))

    all_sub_adj = {'adj_0_0': adj0, 'adj_1_1': adj1, 'adj_2_2': adj2, 'adj_0_1': adj01, 'adj_0_2': adj02}

    node_types = make_node_types(n0, n1, n2)

    features = np.concatenate((node_types, one_hot_labels), axis=1)

    return all_sub_adj, node_types, features


def load_infra():
    fdf0 = pd.read_csv('./data/infra/infra.feat0', delimiter=' ', header=None)
    fdf1 = pd.read_csv('./data/infra/infra.feat1', delimiter=' ', header=None)
    fdf2 = pd.read_csv('./data/infra/infra.feat2', delimiter=' ', header=None)

    labels = np.concatenate((fdf0.values[:, -1], fdf1.values[:, -1], fdf2.values[:, -1]), axis=0)
    one_hot_labels = make_one_hot(labels)

    df0 = pd.read_csv('./data/infra/infra.adj0')
    df1 = pd.read_csv('./data/infra/infra.adj1')
    df2 = pd.read_csv('./data/infra/infra.adj2')

    df01 = pd.read_csv('./data/infra/infra.bet0_1')
    df02 = pd.read_csv('./data/infra/infra.bet0_2')
    df12 = pd.read_csv('./data/infra/infra.bet1_2')

    adj0_indices = get_indices(df0)
    adj1_indices = get_indices(df1)
    adj2_indices = get_indices(df2)

    n0 = np.max(adj0_indices, axis=0)[0] + 1
    n1 = np.max(adj1_indices, axis=0)[0] + 1
    n2 = np.max(adj2_indices, axis=0)[0] + 1

    adj0 = make_sparse_matrix(adj0_indices, (n0, n0))
    adj1 = make_sparse_matrix(adj1_indices, (n1, n1))
    adj2 = make_sparse_matrix(adj2_indices, (n2, n2))

    adj01_indices = get_indices(df01)
    adj02_indices = get_indices(df02)
    adj12_indices = get_indices(df12)

    adj01 = make_sparse_matrix(adj01_indices, (n0, n1))
    adj02 = make_sparse_matrix(adj02_indices, (n0, n2))
    adj12 = make_sparse_matrix(adj12_indices, (n1, n2))

    all_sub_adj = {'adj_0_0': adj0, 'adj_1_1': adj1, 'adj_2_2': adj2, 'adj_0_1': adj01, 'adj_0_2': adj02,
                   'adj_1_2': adj12}

    node_types = make_node_types(n0, n1, n2)

    features = np.concatenate((node_types, one_hot_labels), axis=1)

    return all_sub_adj, node_types, features


def selection(mat, num_val, num_test, diagonal=False):
    if diagonal:
        mat_triu = sp.triu(mat)
    else:
        mat_triu = mat
    mat_tuple = sparse_to_tuple(mat_triu)
    elements = mat_tuple[0]

    all_edge_idx = list(range(elements.shape[0]))
    np.random.shuffle(all_edge_idx)

    val_edge_idx = all_edge_idx[:num_val]
    val_edges = elements[val_edge_idx]

    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = elements[test_edge_idx]

    train_edges = np.delete(elements, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    return train_edges, val_edges, test_edges


def masking(true_indices, false_indices, n_total, n_edges, shape):
    true_data = np.ones(true_indices.shape[0], dtype=float) * (float(n_total) / n_edges)
    true_mask = sp.csr_matrix((true_data, (true_indices[:, 0], true_indices[:, 1])), shape=shape)

    false_data = np.ones(false_indices.shape[0], dtype=float) * (float(n_total) / (n_total - n_edges))
    false_mask = sp.csr_matrix((false_data, (false_indices[:, 0], false_indices[:, 1])), shape=shape)

    final_mask = true_mask + false_mask

    return final_mask


def load_train_val_test(adj, diagonal=False):
    complement = sp.csr_matrix(np.ones(shape=adj.shape, dtype=int)) - adj

    if diagonal:
        non_edges = complement - sp.dia_matrix((complement.diagonal()[np.newaxis, :], [0]), shape=complement.shape)
        edges = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    else:
        non_edges = complement
        edges = adj
    non_edges.eliminate_zeros()
    edges.eliminate_zeros()
    n_edges = edges.nonzero()[0].shape[0]
    n_total = edges.nonzero()[0].shape[0] + non_edges.nonzero()[0].shape[0]

    num_val = int(edges.nonzero()[0].shape[0] * 0.05)
    num_test = int(edges.nonzero()[0].shape[0] * 0.1)
    train_edges, val_edges, test_edges = selection(edges, num_val, num_test, diagonal=diagonal)
    train_false_edges, val_false_edges, test_false_edges = selection(non_edges, num_val, num_test, diagonal=diagonal)

    train_mask = masking(train_edges, train_false_edges, n_total, n_edges, shape=adj.shape)
    val_mask = masking(val_edges, val_false_edges, n_total, n_edges, shape=adj.shape)
    test_mask = masking(test_edges, test_false_edges, n_total, n_edges, shape=adj.shape)

    train_mask = train_mask.todense()
    val_mask = val_mask.todense()
    test_mask = test_mask.todense()

    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    if diagonal:
        adj_train = adj_train + adj_train.T

    return adj_train, train_mask, val_mask, test_mask


def load_train_val_test2(all_sub_adj):
    sub_adj_train = {}
    sub_train_mask = {}
    sub_val_mask = {}
    sub_test_mask = {}
    for key, adj in all_sub_adj.items():
        sub_adj_train[key], sub_train_mask[key], sub_val_mask[key], sub_test_mask[key] = load_train_val_test(adj,
                                                                                                             diagonal=(
                                                                                                             key[-1] ==
                                                                                                             key[-3]))

    return sub_adj_train, sub_train_mask, sub_val_mask, sub_test_mask
