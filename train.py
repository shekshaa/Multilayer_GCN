import tensorflow as tf
from gcn.utils import preprocess_adj, chebyshev_polynomials, sparse_to_tuple
from utils import load_train_val_test, load_aminer, load_infra
from models import Model

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'kipf_gcn', 'cheby_gcn'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('fc_dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('gc_dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('lmbda', 1e-2, 'Weight for type classification loss term')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

features, adj, node_types = load_infra()
n_nodes = features.shape[0]
n_features = features.shape[1]
n_types = node_types.shape[1]

adj_orig = adj.todense()
train_adj, train_mask, val_mask, test_mask = load_train_val_test(adj)
features = sparse_to_tuple(features)

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
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(n_supports)],
    'features': tf.sparse_placeholder(tf.float32),
    'edge_labels': tf.placeholder(tf.int32, shape=[n_nodes, n_nodes]),
    'edge_mask': tf.placeholder(tf.int32, shape=[n_nodes, n_nodes]),
    'node_types': tf.placeholder(tf.int32, shape=[n_nodes, n_types]),
    'gc_dropout': tf.placeholder_with_default(0., shape=()),
    'fc_dropout': tf.placeholder_with_default(0., shape=()),
    'is_train': tf.placeholder(tf.int32),
    'num_features_nonzero': tf.placeholder(tf.int32)
}

print(features[2][1])
model = Model(name='Multilayer_GCN', placeholders=placeholders, num_features=features[2][1], num_nodes=adj.shape[0])

print("Model Created!")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed_dict = dict()
    feed_dict[placeholders['features']] = features
    feed_dict[placeholders['edge_labels']] = adj_orig
    feed_dict[placeholders['node_types']] = node_types
    feed_dict[placeholders['num_features_nonzero']] = features[1].shape

    for i in range(n_supports):
        feed_dict[placeholders['support'][i]] = support[i]

    for epoch in range(FLAGS.epochs):
        feed_dict[placeholders['is_train']] = 1
        feed_dict[placeholders['gc_dropout']] = FLAGS.gc_dropout
        feed_dict[placeholders['fc_dropout']] = FLAGS.fc_dropout
        feed_dict[placeholders['edge_mask']] = train_mask

        __, train_type_acc, train_edge_acc, train_loss = sess.run(
            [model.opt, model.type_acc, model.edge_acc, model.total_loss], feed_dict=feed_dict)

        feed_dict[placeholders['is_train']] = 0
        feed_dict[placeholders['gc_dropout']] = 0.
        feed_dict[placeholders['fc_dropout']] = 0.
        feed_dict[placeholders['edge_mask']] = val_mask

        val_type_acc, val_edge_acc, val_loss = sess.run([model.type_acc, model.edge_acc, model.total_loss],
                                                        feed_dict=feed_dict)

        feed_dict[placeholders['is_train']] = 0
        feed_dict[placeholders['gc_dropout']] = 0.
        feed_dict[placeholders['fc_dropout']] = 0.
        feed_dict[placeholders['edge_mask']] = test_mask

        test_type_acc, test_edge_acc, test_loss = sess.run([model.type_acc, model.edge_acc, model.total_loss],
                                                           feed_dict=feed_dict)

        print('Epoch {}'.format(epoch + 1))
        print('Train: loss={:.3f}, type_acc={:.3f}, edge_acc={:.3f}'.format(train_loss, train_type_acc, train_edge_acc))
        print('Val: loss={:.3f}, type_acc={:.3f}, edge_acc={:.3f}'.format(val_loss, val_type_acc, val_edge_acc))
        print('Test: loss={:.3f}, type_acc={:.3f}, edge_acc={:.3f}'.format(test_loss, test_type_acc, test_edge_acc))
        print('--------')
