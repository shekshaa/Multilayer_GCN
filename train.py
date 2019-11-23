import tensorflow as tf
import numpy as np
from gcn.utils import preprocess_adj, chebyshev_polynomials, sparse_to_tuple
import scipy.sparse as sp
from datetime import datetime
from utils import load_train_val_test2, load_infra, load_aminer
from models import Model

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'kipf_gcn', 'cheby_gcn'
flags.DEFINE_string('dataset', 'aminer', 'Dataset string.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 4, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 128, 'Number of units in hidden layer 3.')
flags.DEFINE_float('fc_dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('gc_dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('lmbda', 0., 'Weight for type classification loss term')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

if FLAGS.dataset == 'infra':
    all_sub_adj, node_types, features = load_infra()
    train_adj, train_mask, val_mask, test_mask = load_train_val_test2(all_sub_adj)
    r0 = sp.hstack((train_adj['adj_0_0'], train_adj['adj_0_1'], train_adj['adj_0_2']), format="csr")
    r1 = sp.hstack((train_adj['adj_0_1'].transpose(), train_adj['adj_1_1'], train_adj['adj_1_2']), format="csr")
    r2 = sp.hstack((train_adj['adj_0_2'].transpose(), train_adj['adj_1_2'].transpose(), train_adj['adj_2_2']),
                   format="csr")
    super_mask = [[1, 1, 1], [0, 1, 1], [0, 0, 1]]
else:
    all_sub_adj, node_types, features = load_aminer()
    train_adj, train_mask, val_mask, test_mask = load_train_val_test2(all_sub_adj)
    n2 = train_adj['adj_0_2'].shape[1]
    n1 = train_adj['adj_0_1'].shape[1]
    empty_mat = sp.csr_matrix(np.zeros(shape=(n1, n2)))
    r0 = sp.hstack((train_adj['adj_0_0'], train_adj['adj_0_1'], train_adj['adj_0_2']), format="csr")
    r1 = sp.hstack((train_adj['adj_0_1'].transpose(), train_adj['adj_1_1'], empty_mat), format="csr")
    r2 = sp.hstack((train_adj['adj_0_2'].transpose(), empty_mat.transpose(), train_adj['adj_2_2']), format="csr")
    super_mask = [[1, 1, 1], [0, 1, 0], [0, 0, 1]]

train_adj = sp.vstack((r0, r1, r2))
n_nodes = train_adj.shape[0]
n_features = features.shape[1]
n_types = node_types.shape[1]

if FLAGS.model == 'gcn':
    support = [preprocess_adj(train_adj)]
    n_supports = 1
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(train_adj, FLAGS.max_degree)
    n_supports = 1 + FLAGS.max_degree
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

print('Supports Created!')

placeholders = {
    'features': tf.placeholder(tf.float32),
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(n_supports)],
    'edge_labels': {key: tf.placeholder(tf.int32) for key, __ in all_sub_adj.items()},
    'edge_mask': {key: tf.sparse_placeholder(tf.float32) for key, ___ in train_mask.items()},
    'node_types': tf.placeholder(tf.int32, shape=[n_nodes, n_types]),
    'gc_dropout': tf.placeholder_with_default(0., shape=()),
    'fc_dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
}

model = Model(name='Multilayer_GCN',
              placeholders=placeholders,
              num_nodes=train_adj.shape[0],
              num_features=n_features,
              super_mask=super_mask)

print("Model Created!")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=tf.ConfigProto())

sess.run(tf.global_variables_initializer())

now = datetime.now()
now_time = now.time()
save_path = str(now.date()) + "_" + str(now_time.hour) + ":" + str(now_time.minute) + ":" + str(now_time.second)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(logdir='./log/{}/'.format(save_path))

feed_dict = dict()
feed_dict[placeholders['features']] = features
feed_dict[placeholders['node_types']] = node_types
feed_dict[placeholders['num_features_nonzero']] = 0.
feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
feed_dict.update({placeholders['edge_labels'][key]: value.todense() for key, value in all_sub_adj.items()})

for epoch in range(FLAGS.epochs):
    feed_dict[placeholders['gc_dropout']] = FLAGS.gc_dropout
    feed_dict[placeholders['fc_dropout']] = FLAGS.fc_dropout
    feed_dict.update({placeholders['edge_mask'][key]: sparse_to_tuple(value) for key, value in train_mask.items()})

    sess.run(model.opt, feed_dict=feed_dict)
    w, tmp, pred = sess.run([model.w['0_1'], model.tmp2[1], model.edge_prediction[1]], feed_dict=feed_dict)
    print(w)
    print(tmp)
    print(np.sum(pred == 1))
    # print(nm_loss)

    summary, train_type_acc, train_edge_f1, train_loss = sess.run(
        [merged_summary, model.type_acc, model.precision, model.total_loss], feed_dict=feed_dict)

    writer.add_summary(summary, global_step=epoch + 1)

    feed_dict[placeholders['gc_dropout']] = 0.
    feed_dict[placeholders['fc_dropout']] = 0.
    feed_dict.update({placeholders['edge_mask'][key]: sparse_to_tuple(value) for key, value in val_mask.items()})

    val_type_acc, val_edge_f1, val_loss = sess.run([model.type_acc, model.precision, model.total_loss],
                                                   feed_dict=feed_dict)

    # tmp = sess.run([model.tmp2[1]], feed_dict=feed_dict)
    # print(tmp)

    feed_dict[placeholders['gc_dropout']] = 0.
    feed_dict[placeholders['fc_dropout']] = 0.
    feed_dict.update({placeholders['edge_mask'][key]: sparse_to_tuple(value) for key, value in test_mask.items()})

    test_type_acc, test_edge_f1, test_loss = sess.run([model.type_acc, model.precision, model.total_loss],
                                                      feed_dict=feed_dict)

    # tmp = sess.run([model.tmp2[1]], feed_dict=feed_dict)
    # print(tmp)

    print('Epoch {}'.format(epoch + 1))
    print('Train: loss={:.3f}, type_acc={:.3f}, edge_f1={:.3f}'.format(train_loss, train_type_acc, train_edge_f1))
    print('Val: loss={:.3f}, type_acc={:.3f}, edge_f1={:.3f}'.format(val_loss, val_type_acc, val_edge_f1))
    print('Test: loss={:.3f}, type_acc={:.3f}, edge_f1={:.3f}'.format(test_loss, test_type_acc, test_edge_f1))
    print('--------')

sess.close()
