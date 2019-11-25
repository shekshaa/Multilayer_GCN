import tensorflow as tf
import numpy as np
from gcn.utils import preprocess_adj, chebyshev_polynomials
import scipy.sparse as sp
from datetime import datetime
from utils import load_train_val_test2, load_infra, load_aminer, visualize_embedding
from models import WeightedAutoencoder

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'kipf_gcn', 'cheby_gcn'
flags.DEFINE_string('dataset', 'infra', 'Dataset string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('node_gc_dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('base_gc_dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('use_weight', 1, 'use w_ij')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('featureless', 1, 'featureless')
flags.DEFINE_float('lmbda', 0., 'Weight for type classification loss term')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

if FLAGS.dataset == 'infra':
    all_sub_adj, node_types, features, labels = load_infra()
    train_adj, train_mask, val_mask, test_mask = load_train_val_test2(all_sub_adj)
    r0 = sp.hstack((train_adj['adj_0_0'], train_adj['adj_0_1'], train_adj['adj_0_2']), format="csr")
    r1 = sp.hstack((train_adj['adj_0_1'].transpose(), train_adj['adj_1_1'], train_adj['adj_1_2']), format="csr")
    r2 = sp.hstack((train_adj['adj_0_2'].transpose(), train_adj['adj_1_2'].transpose(), train_adj['adj_2_2']),
                   format="csr")
    super_mask = [[1, 1, 1], [0, 1, 1], [0, 0, 1]]
else:
    all_sub_adj, node_types, features, labels = load_aminer()
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
    'edge_mask': {key: tf.placeholder(tf.float32) for key, ___ in train_mask.items()},
    'node_types': tf.placeholder(tf.int32, shape=[n_nodes, n_types]),
    'base_gc_dropout': tf.placeholder_with_default(0., shape=()),
    'node_gc_dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
}

model = WeightedAutoencoder(name='Multilayer_GCN',
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

now = datetime.now()
now_time = now.time()
save_path = str(now.date()) + "_" + str(now_time.hour) + ":" + str(now_time.minute) + ":" + str(now_time.second)
train_writer = tf.summary.FileWriter(logdir='./log/{}/train/'.format(save_path))
val_writer = tf.summary.FileWriter(logdir='./log/{}/val/'.format(save_path))

feed_dict = dict()
feed_dict[placeholders['features']] = features
feed_dict[placeholders['node_types']] = node_types
feed_dict[placeholders['num_features_nonzero']] = 0.
feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
feed_dict.update({placeholders['edge_labels'][key]: value.todense() for key, value in all_sub_adj.items()})

for epoch in range(FLAGS.epochs):
    feed_dict[placeholders['base_gc_dropout']] = FLAGS.base_gc_dropout
    feed_dict[placeholders['node_gc_dropout']] = FLAGS.node_gc_dropout
    feed_dict.update({placeholders['edge_mask'][key]: value for key, value in train_mask.items()})

    sess.run(model.opt, feed_dict=feed_dict)

    train_summary, train_type_acc, train_edge_f1, train_loss = sess.run(
        [model.summary1, model.type_acc, model.precision, model.total_loss], feed_dict=feed_dict)

    train_writer.add_summary(train_summary, global_step=epoch + 1)

    feed_dict[placeholders['base_gc_dropout']] = 0.
    feed_dict[placeholders['node_gc_dropout']] = 0.
    feed_dict.update({placeholders['edge_mask'][key]: value for key, value in val_mask.items()})

    val_summary1, val_summary2, val_type_acc, val_edge_f1, val_loss = sess.run(
        [model.summary1, model.summary2, model.type_acc,
         model.precision, model.total_loss],
        feed_dict=feed_dict)

    val_writer.add_summary(val_summary1, global_step=epoch + 1)
    val_writer.add_summary(val_summary2, global_step=epoch + 1)
    print('Epoch {}'.format(epoch + 1))
    if FLAGS.lmbda > 0:
        print('Train: loss={:.3f}, type_acc={:.3f}'.format(train_loss, train_type_acc))
        print('Val: loss={:.3f}, type_acc={:.3f}, edge_f1={:.3f}'.format(val_loss, val_type_acc, val_edge_f1))
    else:
        print('Train: loss={:.3f}'.format(train_loss))
        print('Val: loss={:.3f}, edge_f1={:.3f}'.format(val_loss, val_edge_f1))
    print('--------')

feed_dict[placeholders['base_gc_dropout']] = 0.
feed_dict[placeholders['node_gc_dropout']] = 0.
feed_dict.update({placeholders['edge_mask'][key]: value for key, value in test_mask.items()})

test_type_acc, test_edge_f1, test_loss = sess.run([model.type_acc, model.precision, model.total_loss],
                                                  feed_dict=feed_dict)
print('Test: loss={:.3f}, type_acc={:.3f}, edge_f1={:.3f}'.format(test_loss, test_type_acc, test_edge_f1))
feed_dict = dict()

feed_dict[placeholders['features']] = features
feed_dict[placeholders['num_features_nonzero']] = 0.
feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
embedding = sess.run(model.h2, feed_dict=feed_dict)
visualize_embedding(embedding, labels)

sess.close()
