import tensorflow as tf
from gcn.layers import GraphConvolution, Dense
from gcn.inits import glorot

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, name, placeholders, num_features, num_nodes, activation=tf.nn.relu, bias=True):
        self.name = name

        # feature variables
        self.features = placeholders['features']
        self.n_nodes = num_nodes
        self.input_dim = num_features
        self.num_features_nonzero = placeholders['num_features_nonzero']

        # adjacency matrix
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
        self.bias = bias
        self.is_train = placeholders['is_train']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.w = {}
        self.layers = []
        self.node_type_logits = None
        self.final_edge_logits = None
        self.node_type_module_input = None
        self.edge_module_input = None
        self.type_acc = 0
        self.type_loss = 0
        self.edge_acc = 0
        self.edge_loss = 0
        self.total_loss = 0

        self.build()
        self.loss()
        self.acc()

        self.opt = self.optimizer.minimize(self.total_loss)

    def build(self):
        layer_placeholders = {
            'support': self.support,
            'dropout': self.gc_dropout,
            'num_features_nonzero': self.num_features_nonzero
        }

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=layer_placeholders,
                                            dropout=True,
                                            sparse_inputs=True,
                                            act=self.activation,
                                            bias=self.bias,
                                            logging=True))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=layer_placeholders,
                                            dropout=True,
                                            sparse_inputs=False,
                                            act=self.activation,
                                            bias=self.bias,
                                            logging=True))

        h1 = self.layers[0](self.features)
        h2 = self.layers[1](h1)

        type_fd_placeholders = {
            'dropout': self.fc_dropout,
            'num_features_nonzero': self.num_features_nonzero
        }
        self.node_type_module_input = h2

        self.layers.append(Dense(input_dim=FLAGS.hidden2,
                                 output_dim=self.n_types,
                                 placeholders=type_fd_placeholders,
                                 sparse_inputs=False,
                                 dropout=True,
                                 act=lambda x: x,
                                 bias=self.bias,
                                 logging=True))

        self.node_type_logits = self.layers[2](self.node_type_module_input)

        self.edge_module_input = h1
        with tf.variable_scope(self.name):
            for i in range(self.n_types):
                for j in range(i, self.n_types):
                    n_features = self.edge_module_input.get_shape().as_list()[1]
                    self.w['{}_{}'.format(i, j)] = glorot(shape=(n_features, n_features),
                                                          name='w_{}_{}'.format(i, j))

        all_edge_prediction = []
        for i in range(self.n_types):
            tmp = []
            for j in range(self.n_types):
                weight = self.w['{}_{}'.format(i, j) if i <= j else '{}_{}'.format(j, i)]
                tmp.append(tf.matmul(tf.matmul(self.edge_module_input, (weight + tf.transpose(weight))),
                                     tf.transpose(self.edge_module_input)))
            all_edge_prediction.append(tf.stack(tmp, axis=-1))
        edge_logits = tf.stack(all_edge_prediction, axis=-1)

        node_type_prediction = tf.one_hot(indices=tf.argmax(self.node_type_logits, 1), depth=self.n_types,
                                          dtype=tf.int32)
        selected_node_type = self.is_train * self.node_types + (1 - self.is_train) * node_type_prediction
        node_types1 = tf.cast(tf.expand_dims(tf.expand_dims(selected_node_type, axis=1), axis=3), dtype=tf.float32)
        node_types2 = tf.cast(tf.expand_dims(tf.expand_dims(selected_node_type, axis=0), axis=2), dtype=tf.float32)
        selection1 = tf.reduce_sum(tf.multiply(edge_logits, node_types1), axis=2, keepdims=True)
        selection2 = tf.reduce_sum(tf.multiply(selection1, node_types2), axis=3, keepdims=True)
        self.final_edge_logits = tf.squeeze(selection2)

    def loss(self):
        self.type_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.node_type_logits,
                                                                                labels=self.node_types))

        self.edge_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.final_edge_logits,
                                                                                labels=tf.cast(self.edge_labels,
                                                                                               dtype=tf.float32)) *
                                        tf.cast(self.edge_mask, dtype=tf.float32))

        l2_reg = 0
        for var in self.layers[0].vars.values():
            l2_reg += tf.nn.l2_loss(var)

        self.total_loss = FLAGS.lmbda * self.type_loss + self.edge_loss + FLAGS.weight_decay * l2_reg

    def acc(self):
        self.type_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.node_type_logits, 1),
                                                        tf.argmax(self.node_types, 1)), dtype=tf.float32))

        edge_prediction = tf.cast(tf.greater_equal(tf.nn.sigmoid(self.final_edge_logits), 0.5), dtype=tf.int32)

        self.edge_acc = tf.reduce_mean(tf.cast(tf.equal(edge_prediction, self.edge_labels), dtype=tf.float32))
