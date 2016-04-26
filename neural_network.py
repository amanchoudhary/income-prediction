from __future__ import print_function
import tensorflow as tf
import numpy as np


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def make_neural_network(train_dataset, train_labels, test_dataset, test_labels, num_labels, features):
    train_dataset, train_labels = randomize(train_dataset, train_labels)
    batch_size = 64
    beta = 0.001

    hidden_nodes1 = 128
    hidden_nodes2 = 128

    graph = tf.Graph()
    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, features))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_test_dataset = tf.constant(test_dataset)

        # new hidden layer 1

        hidden_weights = tf.Variable( tf.truncated_normal([features, hidden_nodes1]) )
        hidden_biases = tf.Variable( tf.zeros([hidden_nodes1]))
        hidden_layer = tf.nn.relu( tf.matmul( tf_train_dataset, hidden_weights) + hidden_biases)

        # add dropout on hidden layer
        keep_prob = tf.placeholder("float")
        hidden_layer_drop = tf.nn.dropout(hidden_layer, keep_prob)

        # new hidden layer 2
        hidden_weights2 = tf.Variable( tf.truncated_normal([hidden_nodes1, hidden_nodes2]) )
        hidden_biases2 = tf.Variable( tf.zeros([hidden_nodes2]))
        hidden_layer2 = tf.nn.relu( tf.matmul( hidden_layer_drop, hidden_weights2) + hidden_biases2)

        # add dropout on hidden layer
        hidden_layer_drop2 = tf.nn.dropout(hidden_layer2, keep_prob)

        # Variables.
        weights = tf.Variable( tf.truncated_normal([hidden_nodes2, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits = tf.matmul(hidden_layer_drop2, weights) + biases

        loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) )
        loss = tf.reduce_mean( loss + beta * tf.nn.l2_loss(weights) )

        # Optimizer.
        global_step = tf.Variable(0)  # count the number of steps taken.
        learnr = tf.placeholder("float")
        learning_rate = tf.train.exponential_decay(learnr, global_step, 100000, 0.95, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step= global_step)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)

        test_relu1 = tf.nn.relu( tf.matmul( tf_test_dataset, hidden_weights) + hidden_biases)
        test_relu2 = tf.nn.relu( tf.matmul( test_relu1, hidden_weights2) + hidden_biases2)

        test_prediction = tf.nn.softmax(tf.matmul(test_relu2, weights) + biases)

        num_steps = 2000


        with tf.Session(graph=graph) as session:
          tf.initialize_all_variables().run()
          print("Initialized")
          for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 1.0, learnr : 0.001}
            _, l, predictions = session.run( [optimizer, loss, train_prediction], feed_dict=feed_dict )
            if (step % 500 == 0):
              print("Minibatch loss at step %d: %f" % (step, l))
              print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))