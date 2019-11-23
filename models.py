import tensorflow as tf
from gcn.layers import GraphConvolution, Dense
from gcn.inits import glorot, zeros

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, name, placeholders, num_nodes, num_features, super_mask, activation=tf.nn.tanh, bias=True):
        self.name = name

        # feature variables
        self.n_features = num_features
        self.n_nodes = num_nodes
        self.num_features_nonzero = placeholders['num_features_nonzero']

        # adjacency matrix
        self.features = placeholders['features']
        self.support = placeholders['support']
        self.edge_labels = placeholders['edge_labels']
        self.edge_mask = placeholders['edge_mask']

        # node type variables
        self.node_types = placeholders['node_types']
        self.n_types = self.node_types.get_shape().as_list()[1]

        # network architectural settings
        self.activation = activation
        self.gc_dropout = placeholders['gc_dropout']
        self.fc_dropout = placeholders['fc_dropout']
        self.super_mask = super_mask
        self.bias = bias

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.w = {}
        self.layers = []
        self.node_type_logits = None
        self.edge_module_input_type = None
        self.edge_logits = {}
        self.node_type_module_input = None
        self.edge_module_input = None
        self.type_acc = 0
        self.type_loss = 0
        self.recall = 0
        self.precision = 0
        self.f1 = 0
        self.total_edge_loss = 0
        self.total_loss = 0

        self.build()
        self.loss()
        self.acc()
        self.precision_recall_f1()

        self.opt = self.optimizer.minimize(self.total_loss)

    def build(self):
        layer_placeholders = {
            'support': self.support,
            'dropout': self.gc_dropout,
            'num_features_nonzero': self.num_features_nonzero
        }

        self.layers.append(GraphConvolution(input_dim=self.n_nodes,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=layer_placeholders,
                                            dropout=False,
                                            act=self.activation,
                                            bias=self.bias,
                                            logging=True,
                                            featureless=True
                                            ))

        # self.layers.append(GraphConvolution(input_dim=self.n_features,
        #                                     output_dim=FLAGS.hidden1,
        #                                     placeholders=layer_placeholders,
        #                                     dropout=False,
        #                                     act=self.activation,
        #                                     bias=self.bias,
        #                                     logging=True,
        #                                     # featureless=True
        #                                     ))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=layer_placeholders,
                                            dropout=True,
                                            act=self.activation,
                                            bias=self.bias,
                                            logging=True))

        # h1 = self.layers[0](self.features)
        h1 = self.layers[0](0.)
        h2 = self.layers[1](h1)

        type_fd_placeholders = {
            'support': self.support,
            'dropout': self.fc_dropout,
            'num_features_nonzero': self.num_features_nonzero
        }
        self.node_type_module_input = h2

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=self.n_types,
                                            placeholders=type_fd_placeholders,
                                            dropout=True,
                                            act=lambda x: x,
                                            bias=self.bias,
                                            logging=True))

        self.node_type_logits = self.layers[2](self.node_type_module_input)

        self.edge_module_input = h2
        with tf.variable_scope(self.name):
            n_features = self.edge_module_input.get_shape().as_list()[1]
            for i in range(self.n_types):
                for j in range(i, self.n_types):
                    if self.super_mask[i][j]:
                        var = glorot(shape=(n_features, n_features), name='w_{}_{}'.format(i, j))
                        tf.summary.histogram(name='w_{}_{}'.format(i, j), values=var)
                        self.w['{}_{}'.format(i, j)] = (var + tf.transpose(var)) / 2.
                        # self.w['{}_{}'.format(i, j)] = var

        self.edge_module_input_type = [tf.boolean_mask(tensor=self.edge_module_input, mask=self.node_types[:, i])
                                       for i in range(self.n_types)]

        self.edge_logits = dict()
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                if self.super_mask[i][j]:
                    weight = self.w['{}_{}'.format(i, j)]
                    self.edge_logits['{}_{}'.format(i, j)] = tf.matmul(tf.matmul(self.edge_module_input_type[i], weight)
                                                                       , tf.transpose(self.edge_module_input_type[j]))

    def loss(self):
        self.type_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.node_type_logits,
                                                                                labels=self.node_types))
        self.total_edge_loss = 0
        self.non_mask_loss = []
        self.edge_loss = []
        self.tmp2 = []
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                if self.super_mask[i][j]:
                    self.non_mask_loss.append(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.edge_logits['{}_{}'
                                                                                      .format(i, j)],
                                                                                      labels=tf.cast(self.edge_labels[
                                                                                                         'adj_{}_{}'.format(
                                                                                                             i,
                                                                                                             j)],
                                                                                                     dtype=tf.float32)))
                    self.edge_loss.append(tf.sparse_tensor_to_dense(
                        self.edge_mask['adj_{}_{}'.format(i, j)].__mul__(self.non_mask_loss[-1])))
                    self.tmp2.append(tf.count_nonzero(self.edge_loss[-1]))
                    tmp = tf.reduce_mean(self.edge_loss[-1])
                    self.total_edge_loss += tmp

        l2_reg = 0
        for var in self.layers[0].vars.values():
            l2_reg += tf.nn.l2_loss(var)

        self.total_loss = FLAGS.lmbda * self.type_loss + self.total_edge_loss + FLAGS.weight_decay * l2_reg

    def acc(self):
        type_correct_predictions = tf.equal(tf.argmax(self.node_type_logits, 1),
                                            tf.argmax(self.node_types, 1))
        self.type_acc = tf.reduce_mean(tf.cast(type_correct_predictions, dtype=tf.float32))

    def precision_recall_f1(self):
        self.edge_prediction = []
        true_positive = true_negative = false_positive = false_negative = 0
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                if self.super_mask[i][j]:
                    if i == j:
                        labels = self.edge_labels['adj_{}_{}'.format(i, j)] + \
                                 tf.transpose(self.edge_labels['adj_{}_{}'.format(i, j)])
                    else:
                        labels = self.edge_labels['adj_{}_{}'.format(i, j)]

                    self.edge_prediction.append(tf.cast(
                        tf.greater_equal(tf.nn.sigmoid(self.edge_logits['{}_{}'.format(i, j)]), 0.5),
                        dtype=tf.int32))
                    edge_prediction = self.edge_prediction[-1]
                    true_positive += tf.count_nonzero(edge_prediction * labels)
                    true_negative += tf.count_nonzero((edge_prediction - 1) * (labels - 1))
                    false_positive += tf.count_nonzero(edge_prediction * (labels - 1))
                    false_negative += tf.count_nonzero((edge_prediction - 1) * labels)

        self.precision = true_positive / (true_positive + false_positive)
        self.recall = true_positive / (true_positive + false_negative)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
