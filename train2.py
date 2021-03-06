import tensorflow as tf
from gcn.utils import preprocess_adj, chebyshev_polynomials
from datetime import datetime
from utils import load_train_val_test2, load_infra, load_aminer
from models import ParallelGCN

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'kipf_gcn', 'cheby_gcn'
flags.DEFINE_string('dataset', 'infra', 'Dataset string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('gc_dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('use_weight', 0, 'use w_ij')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('featureless', 1, 'featureless')
flags.DEFINE_float('lmbda', 0., 'Weight for label classification loss term')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

if FLAGS.dataset == 'infra':
    all_sub_adj, node_types, features, one_hot_labels = load_infra()
    train_adj, train_mask, val_mask, test_mask = load_train_val_test2(all_sub_adj)
    train_adj = [train_adj['adj_{}_{}'.format(0, 0)], train_adj['adj_{}_{}'.format(1, 1)],
                 train_adj['adj_{}_{}'.format(2, 2)]]
    super_mask = [[1, 1, 1], [0, 1, 1], [0, 0, 1]]
else:
    all_sub_adj, node_types, features, one_hot_labels = load_aminer()
    train_adj, train_mask, val_mask, test_mask = load_train_val_test2(all_sub_adj)
    train_adj = [train_adj['adj_{}_{}'.format(0, 0)], train_adj['adj_{}_{}'.format(1, 1)],
                 train_adj['adj_{}_{}'.format(2, 2)]]
    super_mask = [[1, 1, 1], [0, 1, 0], [0, 0, 1]]

n_nodes = [adj.shape[0] for adj in train_adj]

if FLAGS.model == 'gcn':
    support = [[preprocess_adj(adj)] for adj in train_adj]
    n_supports = 1
elif FLAGS.model == 'gcn_cheby':
    support = [chebyshev_polynomials(adj, FLAGS.max_degree) for adj in train_adj]
    n_supports = 1 + FLAGS.max_degree
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

print('Supports Created!')

placeholders = {
    'features': tf.placeholder(tf.float32),
    'support0': [tf.sparse_placeholder(tf.float32) for _ in range(n_supports)],
    'support1': [tf.sparse_placeholder(tf.float32) for _ in range(n_supports)],
    'support2': [tf.sparse_placeholder(tf.float32) for _ in range(n_supports)],
    'edge_labels': {key: tf.placeholder(tf.int32) for key, __ in all_sub_adj.items()},
    'edge_mask': {key: tf.placeholder(tf.float32) for key, ___ in train_mask.items()},
    'gc_dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
}

model = ParallelGCN(name='Multilayer_GCN',
                    placeholders=placeholders,
                    num_nodes=n_nodes,
                    super_mask=super_mask,
                    use_weight=FLAGS.use_weight)

print("Model Created!")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=tf.ConfigProto())

sess.run(tf.global_variables_initializer())

now = datetime.now()
now_time = now.time()
save_path = str(now.date()) + "_" + str(now_time.hour) + ":" + str(now_time.minute) + "_w" + str(FLAGS.use_weight)
train_writer = tf.summary.FileWriter(logdir='./log/Parallel/{}/train/'.format(save_path))
val_writer = tf.summary.FileWriter(logdir='./log/Parallel/{}/val/'.format(save_path))

feed_dict = dict()
feed_dict[placeholders['num_features_nonzero']] = 0.
feed_dict.update({placeholders['support0'][i]: support[0][i] for i in range(len(support[0]))})
feed_dict.update({placeholders['support1'][i]: support[1][i] for i in range(len(support[1]))})
feed_dict.update({placeholders['support2'][i]: support[2][i] for i in range(len(support[2]))})
feed_dict.update({placeholders['edge_labels'][key]: value.todense() for key, value in all_sub_adj.items()})

for epoch in range(FLAGS.epochs):
    feed_dict[placeholders['gc_dropout']] = FLAGS.gc_dropout
    feed_dict.update({placeholders['edge_mask'][key]: value for key, value in train_mask.items()})

    sess.run(model.opt, feed_dict=feed_dict)

    train_summary, train_loss, train_f1 = sess.run([model.summary1, model.total_loss, model.f1], feed_dict=feed_dict)
    train_writer.add_summary(train_summary, global_step=epoch + 1)

    feed_dict[placeholders['gc_dropout']] = 0.
    feed_dict.update({placeholders['edge_mask'][key]: value for key, value in val_mask.items()})

    val_summary1, val_summary2, val_loss, val_f1 = sess.run([model.summary1, model.summary2,
                                                             model.total_loss, model.f1], feed_dict=feed_dict)

    val_writer.add_summary(val_summary1, global_step=epoch + 1)
    val_writer.add_summary(val_summary2, global_step=epoch + 1)

    print('Epoch {}'.format(epoch + 1))
    print('Train: loss={:.3f}'.format(train_loss))
    print('Val: loss={:.3f}, edge_f1={:.3f}'.format(val_loss, val_f1))

feed_dict[placeholders['gc_dropout']] = 0.
feed_dict.update({placeholders['edge_mask'][key]: value for key, value in test_mask.items()})

test_loss, test_f1 = sess.run([model.total_loss, model.f1], feed_dict=feed_dict)

print('Test: loss={:.3f}, edge_f1={:.3f}'.format(test_loss, test_f1))
print('--------')
sess.close()
