import tensorflow as tf
from gcn.layers import GraphConvolution
from gcn.inits import glorot

flags = tf.app.flags
FLAGS = flags.FLAGS


class Seq_EF_ML_GCN(object):
    def __init__(self, name, placeholders, num_nodes, super_mask, use_weight,
                 featureless=True, activation=tf.nn.relu, bias=True):
        self.name = name

        # feature variables
        self.n_nodes = num_nodes
        self.n_features = num_nodes if featureless else placeholders['features'].get_shape().as_list()[1]
        self.features = 0. if featureless else placeholders['features']
        self.num_features_nonzero = placeholders['num_features_nonzero']

        # adjacency matrix
        self.EFGCN_support = placeholders['support']
        self.MLGCN_support = [placeholders['support0'], placeholders['support1'], placeholders['support2']]
        self.edge_labels = placeholders['edge_labels']
        self.edge_mask = placeholders['edge_mask']

        # node type variables
        self.node_types = placeholders['node_types']
        self.n_types = len(self.MLGCN_support)

        # network architectural settings
        self.use_weight = use_weight
        self.activation = activation
        self.EFGCN_dropout = placeholders['EFGCN_dropout']
        self.MLGCN_dropout = placeholders['MLGCN_dropout']
        self.super_mask = super_mask
        self.featureless = featureless
        self.bias = bias

        # initialization of model variables
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.h1 = None
        self.h2 = None
        self.w = {}
        self.EFGCN_layers = []
        self.MLGCN_layers = {}
        self.EFGCN_output_separated = []
        self.MLGCN_output = []
        self.edge_logits = {}
        self.recall = 0
        self.precision = 0
        self.f1 = 0
        self.total_edge_loss = 0
        self.total_loss = 0

        # build inference mode and loss and accuracy
        self.build()
        self.loss()
        # self.acc()
        self.precision_recall_f1()
        self.summary1, self.summary2 = self.create_summary()

        self.opt = self.optimizer.minimize(self.total_loss)

    def build(self):
        EFGCN_placeholders = {
            'support': self.EFGCN_support,
            'dropout': self.EFGCN_dropout,
            'num_features_nonzero': self.num_features_nonzero
        }

        self.EFGCN_layers.append(GraphConvolution(input_dim=self.n_features,
                                                  output_dim=FLAGS.hidden1,
                                                  placeholders=EFGCN_placeholders,
                                                  dropout=False,
                                                  act=self.activation,
                                                  bias=self.bias,
                                                  logging=True,
                                                  featureless=self.featureless
                                                  ))

        self.EFGCN_layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                  output_dim=FLAGS.hidden2,
                                                  placeholders=EFGCN_placeholders,
                                                  dropout=True,
                                                  act=self.activation,
                                                  bias=self.bias,
                                                  logging=True))

        self.h1 = self.EFGCN_layers[0](self.features)
        self.h2 = self.EFGCN_layers[1](self.h1)

        EFGCN_output = self.h2
        self.EFGCN_output_separated = [tf.boolean_mask(tensor=EFGCN_output, mask=self.node_types[:, i])
                                       for i in range(self.n_types)]

        for i in range(self.n_types):
            MLGCN_placeholders = {
                'support': self.MLGCN_support[i],
                'dropout': self.MLGCN_dropout,
                'num_features_nonzero': self.num_features_nonzero
            }
            self.MLGCN_layers['{}'.format(i)] = GraphConvolution(input_dim=FLAGS.hidden2,
                                                                 output_dim=FLAGS.hidden3,
                                                                 placeholders=MLGCN_placeholders,
                                                                 dropout=True,
                                                                 act=lambda x: x,
                                                                 bias=self.bias,
                                                                 logging=True)

            self.MLGCN_output.append(self.MLGCN_layers['{}'.format(i)](self.EFGCN_output_separated[i]))

        with tf.variable_scope(self.name):
            n_features = FLAGS.hidden3
            for i in range(self.n_types):
                for j in range(i, self.n_types):
                    if self.super_mask[i][j]:
                        var = glorot(shape=(n_features, n_features), name='w_{}_{}'.format(i, j))
                        tf.summary.histogram(name='w_{}_{}'.format(i, j), values=var)
                        self.w['{}_{}'.format(i, j)] = (var + tf.transpose(var)) / 2.

        self.edge_logits = dict()
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                if self.super_mask[i][j]:
                    if self.use_weight:
                        weight = self.w['{}_{}'.format(i, j)]
                        self.edge_logits['{}_{}'.format(i, j)] = tf.matmul(tf.matmul(self.MLGCN_output[i], weight),
                                                                           tf.transpose(self.MLGCN_output[j]))
                    else:
                        self.edge_logits['{}_{}'.format(i, j)] = tf.matmul(self.MLGCN_output[i],
                                                                           tf.transpose(self.MLGCN_output[j]))

    def loss(self):
        self.total_edge_loss = 0
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                if self.super_mask[i][j]:
                    non_mask_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.edge_logits['{}_{}'
                                                                            .format(i, j)],
                                                                            labels=tf.cast(self.edge_labels[
                                                                                               'adj_{}_{}'.format(
                                                                                                   i,
                                                                                                   j)],
                                                                                           dtype=tf.float32))

                    self.total_edge_loss += tf.reduce_mean(self.edge_mask['adj_{}_{}'.format(i, j)] * non_mask_loss)

        l2_reg = 0
        for var in self.EFGCN_layers[0].vars.values():
            l2_reg += tf.nn.l2_loss(var)

        self.total_loss = self.total_edge_loss + FLAGS.weight_decay * l2_reg

    def create_summary(self):
        summary1_list = [tf.summary.scalar(name='total_edge_loss', tensor=self.total_edge_loss),
                         tf.summary.scalar(name='total_loss', tensor=self.total_loss)]

        summary2_list = [tf.summary.scalar(name='precision', tensor=self.precision),
                         tf.summary.scalar(name='recall', tensor=self.recall),
                         tf.summary.scalar(name='F1', tensor=self.f1)]
        return tf.summary.merge(summary1_list), tf.summary.merge(summary2_list)

    # def acc(self):
    #     label_correct_predictions = tf.equal(tf.argmax(self.node_label_logits, 1),
    #                                          tf.argmax(self.node_types, 1))
    #     self.label_acc = tf.reduce_mean(tf.cast(label_correct_predictions, dtype=tf.float32))

    def precision_recall_f1(self):
        true_positive = true_negative = false_positive = false_negative = 0
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                if self.super_mask[i][j]:
                    labels = self.edge_labels['adj_{}_{}'.format(i, j)]
                    edge_prediction = tf.cast(
                        tf.greater_equal(tf.nn.sigmoid(self.edge_logits['{}_{}'.format(i, j)]), 0.5),
                        dtype=tf.int32)
                    mask = self.edge_mask['adj_{}_{}'.format(i, j)]
                    true_positive += tf.count_nonzero(tf.cast(edge_prediction * labels, dtype=tf.float32) * mask)
                    true_negative += tf.count_nonzero(tf.cast((edge_prediction - 1) * (labels - 1),
                                                              dtype=tf.float32) * mask)
                    false_positive += tf.count_nonzero(tf.cast(edge_prediction * (labels - 1), dtype=tf.float32) * mask)
                    false_negative += tf.count_nonzero(tf.cast((edge_prediction - 1) * labels, dtype=tf.float32) * mask)

        self.precision = true_positive / (true_positive + false_positive)
        self.recall = true_positive / (true_positive + false_negative)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)


class Parallel_EF_ML_GCN(object):
    def __init__(self, name, placeholders, num_nodes, super_mask, use_weight, n_nodes_separated,
                 featureless=True, activation=tf.nn.relu, bias=True):
        self.name = name

        # feature variables
        self.n_nodes_separated = n_nodes_separated
        self.n_nodes = num_nodes
        self.n_features = num_nodes if featureless else placeholders['features'].get_shape().as_list()[1]
        self.features = 0. if featureless else placeholders['features']
        self.num_features_nonzero = placeholders['num_features_nonzero']

        # adjacency matrix
        self.EFGCN_support = placeholders['support']
        self.MLGCN_support = [placeholders['support0'], placeholders['support1'], placeholders['support2']]
        self.edge_labels = placeholders['edge_labels']
        self.edge_mask = placeholders['edge_mask']

        # node type variables
        self.node_types = placeholders['node_types']
        self.n_types = len(self.MLGCN_support)

        # network architectural settings
        self.use_weight = use_weight
        self.activation = activation
        self.EFGCN_dropout = placeholders['EFGCN_dropout']
        self.MLGCN_dropout = placeholders['MLGCN_dropout']
        self.super_mask = super_mask
        self.featureless = featureless
        self.bias = bias

        # initialization of model variables
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.h1 = None
        self.h2 = None
        self.w = {}
        self.EFGCN_layers = []
        self.MLGCN_layers = {}
        self.EFGCN_output_separated = []
        self.MLGCN_output = []
        self.MLGCN_hidden = []
        self.aggregated_output = []
        self.edge_logits = {}
        self.recall = 0
        self.precision = 0
        self.f1 = 0
        self.total_edge_loss = 0
        self.total_loss = 0

        # build inference mode and loss and accuracy
        self.build()
        self.loss()
        # self.acc()
        self.precision_recall_f1()
        self.summary1, self.summary2 = self.create_summary()

        self.opt = self.optimizer.minimize(self.total_loss)

    def build(self):
        EFGCN_placeholders = {
            'support': self.EFGCN_support,
            'dropout': self.EFGCN_dropout,
            'num_features_nonzero': self.num_features_nonzero
        }

        self.EFGCN_layers.append(GraphConvolution(input_dim=self.n_features,
                                                  output_dim=FLAGS.hidden1,
                                                  placeholders=EFGCN_placeholders,
                                                  dropout=False,
                                                  act=self.activation,
                                                  bias=self.bias,
                                                  logging=True,
                                                  featureless=self.featureless))

        self.EFGCN_layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                  output_dim=FLAGS.hidden2,
                                                  placeholders=EFGCN_placeholders,
                                                  dropout=True,
                                                  act=self.activation,
                                                  bias=self.bias,
                                                  logging=True))

        self.h1 = self.EFGCN_layers[0](self.features)
        self.h2 = self.EFGCN_layers[1](self.h1)

        EFGCN_output = self.h2
        self.EFGCN_output_separated = [tf.boolean_mask(tensor=EFGCN_output, mask=self.node_types[:, i])
                                       for i in range(self.n_types)]

        for i in range(self.n_types):
            MLGCN_placeholders = {
                'support': self.MLGCN_support[i],
                'dropout': self.MLGCN_dropout,
                'num_features_nonzero': self.num_features_nonzero
            }
            self.MLGCN_layers['layer1_{}'.format(i)] = GraphConvolution(input_dim=self.n_nodes_separated[i],
                                                                        output_dim=FLAGS.hidden1,
                                                                        placeholders=MLGCN_placeholders,
                                                                        dropout=False,
                                                                        act=self.activation,
                                                                        bias=self.bias,
                                                                        logging=True,
                                                                        featureless=self.featureless)

            self.MLGCN_layers['layer2_{}'.format(i)] = GraphConvolution(input_dim=FLAGS.hidden1,
                                                                        output_dim=FLAGS.hidden2,
                                                                        placeholders=MLGCN_placeholders,
                                                                        dropout=True,
                                                                        act=lambda x: x,
                                                                        bias=self.bias,
                                                                        logging=True)

            self.MLGCN_hidden.append(self.MLGCN_layers['layer1_{}'.format(i)](0.))
            self.MLGCN_output.append(self.MLGCN_layers['layer2_{}'.format(i)](self.MLGCN_hidden[-1]))

        if FLAGS.aggregation == 'mean':
            self.aggregated_output = [(self.MLGCN_output[i] + self.EFGCN_output_separated[i]) / 2.
                                      for i in range(self.n_types)]
            n_features = FLAGS.hidden2
        elif FLAGS.aggregation == 'concat':
            self.aggregated_output = [tf.concat(values=(self.MLGCN_output[i], self.EFGCN_output_separated[i]), axis=1)
                                      for i in range(self.n_types)]
            n_features = FLAGS.hidden2 * 2
        else:
            raise Exception

        with tf.variable_scope(self.name):
            for i in range(self.n_types):
                for j in range(i, self.n_types):
                    if self.super_mask[i][j]:
                        var = glorot(shape=(n_features, n_features), name='w_{}_{}'.format(i, j))
                        tf.summary.histogram(name='w_{}_{}'.format(i, j), values=var)
                        self.w['{}_{}'.format(i, j)] = (var + tf.transpose(var)) / 2.

        self.edge_logits = dict()
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                if self.super_mask[i][j]:
                    if self.use_weight:
                        weight = self.w['{}_{}'.format(i, j)]
                        self.edge_logits['{}_{}'.format(i, j)] = tf.matmul(tf.matmul(self.aggregated_output[i], weight),
                                                                           tf.transpose(self.aggregated_output[j]))
                    else:
                        self.edge_logits['{}_{}'.format(i, j)] = tf.matmul(self.aggregated_output[i],
                                                                           tf.transpose(self.aggregated_output[j]))

    def loss(self):
        self.total_edge_loss = 0
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                if self.super_mask[i][j]:
                    non_mask_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.edge_logits['{}_{}'
                                                                            .format(i, j)],
                                                                            labels=tf.cast(self.edge_labels[
                                                                                               'adj_{}_{}'.format(
                                                                                                   i,
                                                                                                   j)],
                                                                                           dtype=tf.float32))

                    self.total_edge_loss += tf.reduce_mean(self.edge_mask['adj_{}_{}'.format(i, j)] * non_mask_loss)

        l2_reg = 0
        for var in self.EFGCN_layers[0].vars.values():
            l2_reg += tf.nn.l2_loss(var)

        self.total_loss = self.total_edge_loss + FLAGS.weight_decay * l2_reg

    def create_summary(self):
        summary1_list = [tf.summary.scalar(name='total_edge_loss', tensor=self.total_edge_loss),
                         tf.summary.scalar(name='total_loss', tensor=self.total_loss)]

        summary2_list = [tf.summary.scalar(name='precision', tensor=self.precision),
                         tf.summary.scalar(name='recall', tensor=self.recall),
                         tf.summary.scalar(name='F1', tensor=self.f1)]
        return tf.summary.merge(summary1_list), tf.summary.merge(summary2_list)

    # def acc(self):
    #     label_correct_predictions = tf.equal(tf.argmax(self.node_label_logits, 1),
    #                                          tf.argmax(self.node_types, 1))
    #     self.label_acc = tf.reduce_mean(tf.cast(label_correct_predictions, dtype=tf.float32))

    def precision_recall_f1(self):
        true_positive = true_negative = false_positive = false_negative = 0
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                if self.super_mask[i][j]:
                    labels = self.edge_labels['adj_{}_{}'.format(i, j)]
                    edge_prediction = tf.cast(
                        tf.greater_equal(tf.nn.sigmoid(self.edge_logits['{}_{}'.format(i, j)]), 0.5),
                        dtype=tf.int32)
                    mask = self.edge_mask['adj_{}_{}'.format(i, j)]
                    true_positive += tf.count_nonzero(tf.cast(edge_prediction * labels, dtype=tf.float32) * mask)
                    true_negative += tf.count_nonzero(tf.cast((edge_prediction - 1) * (labels - 1),
                                                              dtype=tf.float32) * mask)
                    false_positive += tf.count_nonzero(tf.cast(edge_prediction * (labels - 1), dtype=tf.float32) * mask)
                    false_negative += tf.count_nonzero(tf.cast((edge_prediction - 1) * labels, dtype=tf.float32) * mask)

        self.precision = true_positive / (true_positive + false_positive)
        self.recall = true_positive / (true_positive + false_negative)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
