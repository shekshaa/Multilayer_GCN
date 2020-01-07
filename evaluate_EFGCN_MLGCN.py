import tensorflow as tf
import numpy as np
import pandas as pnd
from gcn.utils import preprocess_adj, chebyshev_polynomials
import scipy.sparse as sp
from datetime import datetime
from utils import load_train_val_test2, load_infra, load_aminer
from models import EFGCN_MLGCN

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn', 'Model name.')
flags.DEFINE_float('learning_rate', 0.05, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('EFGCN_dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('MLGCN_dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('use_weight', 1, 'use w_ij')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('featureless', 1, 'featureless')
flags.DEFINE_float('lmbda', 0., 'Weight for label classification loss term')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


def load_data(dataset):
    if dataset == 'infra':
        all_sub_adj, node_types, features, one_hot_labels = load_infra()
    elif dataset == 'aminer':
        all_sub_adj, node_types, features, one_hot_labels = load_aminer()
    else:
        raise Exception

    return all_sub_adj, node_types, features, one_hot_labels


def train(train_adj, separated_train_adj, all_sub_adj, features,
          train_mask, val_mask, test_mask, super_mask,
          node_types, one_hot_labels,
          time_str, r):
    n_nodes = train_adj.shape[0]
    n_types = node_types.shape[1]
    n_labels = one_hot_labels.shape[1]

    if FLAGS.model == 'gcn':
        support = [preprocess_adj(train_adj)]
        separated_support = [[preprocess_adj(adj)] for adj in separated_train_adj]
        n_supports = 1
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(train_adj, FLAGS.max_degree)
        separated_support = [chebyshev_polynomials(adj, FLAGS.max_degree) for adj in separated_train_adj]
        n_supports = 1 + FLAGS.max_degree
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    print('Supports Created!')

    placeholders = {
        'features': tf.placeholder(tf.float32),
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(n_supports)],
        'support0': [tf.sparse_placeholder(tf.float32) for _ in range(n_supports)],
        'support1': [tf.sparse_placeholder(tf.float32) for _ in range(n_supports)],
        'support2': [tf.sparse_placeholder(tf.float32) for _ in range(n_supports)],
        'edge_labels': {key: tf.placeholder(tf.int32) for key, __ in all_sub_adj.items()},
        'edge_mask': {key: tf.placeholder(tf.float32) for key, ___ in train_mask.items()},
        'node_types': tf.placeholder(tf.int32, shape=[n_nodes, n_types]),
        'node_labels': tf.placeholder(tf.int32, shape=[n_nodes, n_labels]),
        'EFGCN_dropout': tf.placeholder_with_default(0., shape=()),
        'MLGCN_dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),
    }

    model = EFGCN_MLGCN(name='EFGCN_MLGCN',
                        placeholders=placeholders,
                        num_nodes=train_adj.shape[0],
                        super_mask=super_mask,
                        use_weight=FLAGS.use_weight,
                        featureless=FLAGS.featureless)

    print("Model Created!")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf.ConfigProto())

    sess.run(tf.global_variables_initializer())

    save_path = str(FLAGS.learning_rate) + "_" + str(FLAGS.hidden1) + "_" + str(FLAGS.hidden3)
    train_writer = tf.summary.FileWriter(logdir='./log/EFGCN_MLGCN/' + time_str + '/' +
                                                save_path + '/train/{}/'.format(r))
    val_writer = tf.summary.FileWriter(logdir='./log/EFGCN_MLGCN/' + time_str + '/' +
                                              save_path + '/val/{}/'.format(r))

    feed_dict = dict()
    feed_dict[placeholders['features']] = features
    feed_dict[placeholders['node_types']] = node_types
    feed_dict[placeholders['node_labels']] = one_hot_labels
    feed_dict[placeholders['num_features_nonzero']] = 0.
    feed_dict.update({placeholders['support0'][i]: separated_support[0][i] for i in range(len(separated_support[0]))})
    feed_dict.update({placeholders['support1'][i]: separated_support[1][i] for i in range(len(separated_support[1]))})
    feed_dict.update({placeholders['support2'][i]: separated_support[2][i] for i in range(len(separated_support[2]))})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['edge_labels'][key]: value.todense() for key, value in all_sub_adj.items()})

    val_edge_f1 = 0
    for epoch in range(FLAGS.epochs):
        feed_dict[placeholders['EFGCN_dropout']] = FLAGS.EFGCN_dropout
        feed_dict[placeholders['MLGCN_dropout']] = FLAGS.MLGCN_dropout
        feed_dict.update({placeholders['edge_mask'][key]: value for key, value in train_mask.items()})

        sess.run(model.opt, feed_dict=feed_dict)

        train_summary, train_loss = sess.run(
            [model.summary1, model.total_loss], feed_dict=feed_dict)

        train_writer.add_summary(train_summary, global_step=epoch + 1)

        feed_dict[placeholders['EFGCN_dropout']] = 0.
        feed_dict[placeholders['MLGCN_dropout']] = 0.
        feed_dict.update({placeholders['edge_mask'][key]: value for key, value in val_mask.items()})

        val_summary1, val_summary2, val_edge_f1, val_loss = sess.run(
            [model.summary1, model.summary2,
             model.f1, model.total_loss],
            feed_dict=feed_dict)

        val_writer.add_summary(val_summary1, global_step=epoch + 1)
        val_writer.add_summary(val_summary2, global_step=epoch + 1)
        print('Epoch {}'.format(epoch + 1))
        print('Train: loss={:.3f}'.format(train_loss))
        print('Val: loss={:.3f}, edge_f1={:.3f}'.format(val_loss, val_edge_f1))
        print('--------')

    feed_dict[placeholders['EFGCN_dropout']] = 0.
    feed_dict[placeholders['MLGCN_dropout']] = 0.
    feed_dict.update({placeholders['edge_mask'][key]: value for key, value in test_mask.items()})

    test_edge_f1, test_loss = sess.run([model.f1, model.total_loss],
                                       feed_dict=feed_dict)

    print('Test: loss={:.3f}, edge_f1={:.3f}'.format(test_loss, test_edge_f1))

    sess.close()
    return val_edge_f1, test_edge_f1


def evaluate(dataset):
    all_sub_adj, node_types, features, one_hot_labels = load_data(dataset)
    if dataset == 'infra':
        train_adj, train_mask, val_mask, test_mask = load_train_val_test2(all_sub_adj)
        r0 = sp.hstack((train_adj['adj_0_0'], train_adj['adj_0_1'], train_adj['adj_0_2']), format="csr")
        r1 = sp.hstack((train_adj['adj_0_1'].transpose(), train_adj['adj_1_1'], train_adj['adj_1_2']), format="csr")
        r2 = sp.hstack((train_adj['adj_0_2'].transpose(), train_adj['adj_1_2'].transpose(), train_adj['adj_2_2']),
                       format="csr")
        separated_train_adj = [train_adj['adj_{}_{}'.format(0, 0)], train_adj['adj_{}_{}'.format(1, 1)],
                               train_adj['adj_{}_{}'.format(2, 2)]]
        super_mask = [[1, 1, 1], [0, 1, 1], [0, 0, 1]]
    else:
        train_adj, train_mask, val_mask, test_mask = load_train_val_test2(all_sub_adj)
        n2 = train_adj['adj_0_2'].shape[1]
        n1 = train_adj['adj_0_1'].shape[1]
        empty_mat = sp.csr_matrix(np.zeros(shape=(n1, n2)))
        r0 = sp.hstack((train_adj['adj_0_0'], train_adj['adj_0_1'], train_adj['adj_0_2']), format="csr")
        r1 = sp.hstack((train_adj['adj_0_1'].transpose(), train_adj['adj_1_1'], empty_mat), format="csr")
        r2 = sp.hstack((train_adj['adj_0_2'].transpose(), empty_mat.transpose(), train_adj['adj_2_2']), format="csr")
        separated_train_adj = [train_adj['adj_{}_{}'.format(0, 0)], train_adj['adj_{}_{}'.format(1, 1)],
                               train_adj['adj_{}_{}'.format(2, 2)]]
        super_mask = [[1, 1, 1], [0, 1, 0], [0, 0, 1]]

    train_adj = sp.vstack((r0, r1, r2))
    num_runs = 10
    learning_rates = [0.005, 0.01, 0.05]
    hidden1 = [64, 32]
    hidden3 = [32, 16]
    val_f1_arr = []
    test_f1_arr = []
    now = datetime.now()
    now_time = now.time()
    time_str = str(now.date()) + "_" + str(now_time.hour) + ":" + str(now_time.minute) + "_w" + str(FLAGS.use_weight)
    for i, l in enumerate(learning_rates):
        FLAGS.learning_rate = l
        val_f1_arr.append([])
        test_f1_arr.append([])
        for j, h1 in enumerate(hidden1):
            for k, h3 in enumerate(hidden3):
                print("Learning rate: {}, hidden1: {}, hidden3: {}".format(l, h1, h3))
                tf.reset_default_graph()
                FLAGS.hidden3 = h1
                FLAGS.hidden3 = h3
                val_f1_list = []
                test_f1_list = []
                for r in range(num_runs):
                    val_f1, test_f1 = train(train_adj, separated_train_adj, all_sub_adj, features,
                                            train_mask, val_mask, test_mask, super_mask,
                                            node_types, one_hot_labels, time_str, r)
                    val_f1_list.append(val_f1 * 100.)
                    test_f1_list.append(test_f1 * 100.)

                val_f1_arr[i].append('{:.2f} ± {:.2f}'.format(np.mean(val_f1_list), np.sqrt(np.var(val_f1_list))))
                test_f1_arr[i].append('{:.2f} ± {:.2f}'.format(np.mean(test_f1_list), np.sqrt(np.var(test_f1_list))))

    columns = [str(h1) + '_' + str(h3) for h1 in hidden1 for h3 in hidden3]
    val_df = pnd.DataFrame(data=val_f1_arr, index=learning_rates, columns=columns, dtype=str)
    test_df = pnd.DataFrame(data=test_f1_arr, index=learning_rates, columns=columns, dtype=str)
    val_df.to_csv(path_or_buf='./log/EFGCN_MLGCN/' + time_str + '/val_f1.csv')
    test_df.to_csv(path_or_buf='./log/EFGCN_MLGCN/' + time_str + '/test_f1.csv')


if __name__ == '__main__':
    evaluate('infra')
