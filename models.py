import tensorflow as tf
from gcn.layers import GraphConvolution
from gcn.inits import glorot

flags = tf.app.flags
FLAGS = flags.FLAGS


class EFGCN_MLGCN(object):
    def __init__(self, name, placeholders, num_nodes, super_mask, use_weight,
                 featureless=True, activation=tf.nn.tanh, bias=True):
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
                                                                 dropout=False,
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

        # for layer in self.MLGCN_layers:
        #     l2_reg += tf.nn.l2_loss(layer.vars.values())

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


class WeightedMixedAutoencoder(object):
    def __init__(self, name, placeholders, num_nodes, super_mask, use_weight,
                 featureless=True, activation=tf.nn.tanh, bias=True):
        self.name = name

        # feature variables
        self.n_nodes = num_nodes
        self.n_features = num_nodes if featureless else placeholders['features'].get_shape().as_list()[1]
        self.features = 0. if featureless else placeholders['features']
        self.num_features_nonzero = placeholders['num_features_nonzero']

        # adjacency matrix
        self.support = placeholders['support']
        self.edge_labels = placeholders['edge_labels']
        self.edge_mask = placeholders['edge_mask']

        # node type variables
        self.node_types = placeholders['node_types']
        self.n_types = self.node_types.get_shape().as_list()[1]

        # network architectural settings
        self.use_weight = use_weight
        self.activation = activation
        self.base_gc_dropout = placeholders['base_gc_dropout']
        self.node_gc_dropout = placeholders['node_gc_dropout']
        self.super_mask = super_mask
        self.featureless = featureless
        self.bias = bias

        # initialization of model variables
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.h1 = None
        self.h2 = None
        self.w = {}
        self.layers = []
        self.node_type_logits = None
        self.node_type_probs = None
        self.edge_module_input_type = None
        self.edge_logits = []
        self.node_type_module_input = None
        self.edge_module_input = None
        self.weighted_embedding = []
        self.type_acc = 0
        self.type_loss = 0
        self.recall = 0
        self.precision = 0
        self.f1 = 0
        self.total_edge_loss = 0
        self.total_loss = 0

        # build inference mode and loss and accuracy
        self.build()
        self.loss()
        self.acc()
        self.precision_recall_f1()
        self.summary1, self.summary2 = self.create_summary()

        self.opt = self.optimizer.minimize(self.total_loss)

    def build(self):
        layer_placeholders = {
            'support': self.support,
            'dropout': self.base_gc_dropout,
            'num_features_nonzero': self.num_features_nonzero
        }

        self.layers.append(GraphConvolution(input_dim=self.n_features,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=layer_placeholders,
                                            dropout=False,
                                            act=self.activation,
                                            bias=self.bias,
                                            logging=True,
                                            featureless=self.featureless
                                            ))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=layer_placeholders,
                                            dropout=True,
                                            act=self.activation,
                                            bias=self.bias,
                                            logging=True))

        self.h1 = self.layers[0](self.features)
        self.h2 = self.layers[1](self.h1)

        type_fd_placeholders = {
            'support': self.support,
            'dropout': self.node_gc_dropout,
            'num_features_nonzero': self.num_features_nonzero
        }
        self.node_type_module_input = self.h2

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=self.n_types,
                                            placeholders=type_fd_placeholders,
                                            dropout=True,
                                            act=lambda x: x,
                                            bias=self.bias,
                                            logging=True))

        self.node_type_logits = self.layers[2](self.node_type_module_input)
        self.node_type_probs = tf.nn.softmax(self.node_type_logits)
        print(self.node_type_probs.shape)

        self.edge_module_input = self.h2
        with tf.variable_scope(self.name):
            n_features = self.edge_module_input.get_shape().as_list()[1]
            for i in range(self.n_types):
                for j in range(i, self.n_types):
                    if self.super_mask[i][j]:
                        # var = glorot(shape=(n_features, n_features), name='w_{}_{}'.format(i, j))
                        # self.w['{}_{}'.format(i, j)] = (var + tf.transpose(var)) / 2.
                        var = tf.squeeze(glorot(shape=(n_features, 1), name='w_{}_{}'.format(i, j)))
                        tf.summary.histogram(name='w_{}_{}'.format(i, j), values=var)
                        self.w['{}_{}'.format(i, j)] = tf.matrix_diag(var)

        for i in range(self.n_types):
            self.weighted_embedding.append(
                tf.multiply(self.edge_module_input, tf.expand_dims(self.node_type_probs[:, i], axis=1)))

        print(self.weighted_embedding)

        for i in range(self.n_types):
            for j in range(i, self.n_types):
                if self.super_mask[i][j]:
                    if self.use_weight:
                        weight = self.w['{}_{}'.format(i, j)]
                        self.edge_logits.append(tf.matmul(
                            tf.matmul(self.weighted_embedding[i], weight), tf.transpose(self.weighted_embedding[j])))
                    else:
                        self.edge_logits.append(tf.matmul(self.weighted_embedding[i],
                                                          tf.transpose(self.weighted_embedding[j])))
        self.edge_logits = tf.add_n(self.edge_logits)
        print(self.edge_logits)

    def loss(self):
        self.type_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.node_type_logits,
                                                                                labels=self.node_types))
        self.total_edge_loss = 0
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                if self.super_mask[i][j]:
                    filtered_logits_i = tf.boolean_mask(tensor=self.edge_logits, mask=self.node_types[:, i])
                    filtered_logits_j = tf.boolean_mask(tensor=tf.transpose(filtered_logits_i),
                                                        mask=self.node_types[:, j])
                    filtered_logits = tf.transpose(filtered_logits_j)
                    non_mask_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits,
                                                                            labels=tf.cast(self.edge_labels[
                                                                                               'adj_{}_{}'.format(i,
                                                                                                                  j)],
                                                                                           dtype=tf.float32))

                    self.total_edge_loss += tf.reduce_mean(self.edge_mask['adj_{}_{}'.format(i, j)] * non_mask_loss)

        l2_reg = 0
        for var in self.layers[0].vars.values():
            l2_reg += tf.nn.l2_loss(var)

        self.total_loss = FLAGS.lmbda * self.type_loss + self.total_edge_loss + FLAGS.weight_decay * l2_reg

    def create_summary(self):
        summary1_list = [tf.summary.scalar(name='total_edge_loss', tensor=self.total_edge_loss),
                         tf.summary.scalar(name='total_loss', tensor=self.total_loss),
                         tf.summary.scalar(name='node_label_loss', tensor=self.type_loss),
                         tf.summary.scalar(name='node_label_acc', tensor=self.type_acc)]
        summary2_list = [tf.summary.scalar(name='precision', tensor=self.precision),
                         tf.summary.scalar(name='recall', tensor=self.recall),
                         tf.summary.scalar(name='F1', tensor=self.f1)]
        return tf.summary.merge(summary1_list), tf.summary.merge(summary2_list)

    def acc(self):
        label_correct_predictions = tf.equal(tf.argmax(self.node_type_logits, 1),
                                             tf.argmax(self.node_types, 1))
        self.type_acc = tf.reduce_mean(tf.cast(label_correct_predictions, dtype=tf.float32))

    def precision_recall_f1(self):
        true_positive = true_negative = false_positive = false_negative = 0
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                if self.super_mask[i][j]:
                    labels = self.edge_labels['adj_{}_{}'.format(i, j)]
                    filtered_logits_i = tf.boolean_mask(tensor=self.edge_logits, mask=self.node_types[:, i])
                    filtered_logits_j = tf.boolean_mask(tensor=tf.transpose(filtered_logits_i),
                                                        mask=self.node_types[:, j])
                    filtered_logits = tf.transpose(filtered_logits_j)
                    edge_prediction = tf.cast(
                        tf.greater_equal(tf.nn.sigmoid(filtered_logits), 0.5),
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


class ParallelGCN(object):
    def __init__(self, name, placeholders, num_nodes, super_mask, use_weight, activation=tf.nn.tanh, bias=True):
        self.name = name

        # feature variables
        self.n_nodes = num_nodes
        self.num_features_nonzero = placeholders['num_features_nonzero']

        # adjacency matrix
        self.support = [placeholders['support0'], placeholders['support1'], placeholders['support2']]
        self.edge_labels = placeholders['edge_labels']
        self.edge_mask = placeholders['edge_mask']

        # node type variables
        self.n_types = len(self.support)

        # network architectural settings
        self.use_weight = use_weight
        self.activation = activation
        self.gc_dropout = placeholders['gc_dropout']
        self.super_mask = super_mask
        self.bias = bias

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.w = {}
        self.layers = {}
        self.h1 = {}
        self.h2 = {}
        self.edge_module_input_type = None
        self.edge_logits = {}
        self.edge_module_input = None
        self.recall = 0
        self.precision = 0
        self.f1 = 0
        self.total_edge_loss = 0
        self.total_loss = 0

        self.build()
        self.loss()
        self.precision_recall_f1()
        self.summary1, self.summary2 = self.create_summary()

        self.opt = self.optimizer.minimize(self.total_loss)

    def build(self):
        for i in range(self.n_types):
            type_placeholders = {
                'support': self.support[i],
                'dropout': self.gc_dropout,
                'num_features_nonzero': self.num_features_nonzero
            }
            layers = [GraphConvolution(input_dim=self.n_nodes[i],
                                       output_dim=FLAGS.hidden1,
                                       placeholders=type_placeholders,
                                       dropout=False,
                                       act=self.activation,
                                       bias=self.bias,
                                       logging=True,
                                       featureless=True
                                       ), GraphConvolution(input_dim=FLAGS.hidden1,
                                                           output_dim=FLAGS.hidden2,
                                                           placeholders=type_placeholders,
                                                           dropout=False,
                                                           act=self.activation,
                                                           bias=self.bias,
                                                           logging=True,
                                                           )]

            self.layers['{}'.format(i)] = layers
            self.h1['{}'.format(i)] = layers[0](0.)
            self.h2['{}'.format(i)] = layers[1](self.h1['{}'.format(i)])

        with tf.variable_scope(self.name):
            n_features = FLAGS.hidden2
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
                        self.edge_logits['{}_{}'.format(i, j)] = tf.matmul(tf.matmul(self.h2['{}'.format(i)], weight)
                                                                           , tf.transpose(self.h2['{}'.format(j)]))
                    else:
                        self.edge_logits['{}_{}'.format(i, j)] = tf.matmul(self.h2['{}'.format(i)],
                                                                           tf.transpose(self.h2['{}'.format(j)]))

    def create_summary(self):
        summary1_list = [tf.summary.scalar(name='total_edge_loss', tensor=self.total_edge_loss),
                         tf.summary.scalar(name='total_loss', tensor=self.total_loss)]

        summary2_list = [tf.summary.scalar(name='precision', tensor=self.precision),
                         tf.summary.scalar(name='recall', tensor=self.recall),
                         tf.summary.scalar(name='F1', tensor=self.f1)]
        return tf.summary.merge(summary1_list), tf.summary.merge(summary2_list)

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
        for _, layers in self.layers.items():
            for var in layers[0].vars.values():
                l2_reg += tf.nn.l2_loss(var)

        self.total_loss = self.total_edge_loss + FLAGS.weight_decay * l2_reg

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
