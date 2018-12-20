import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from gcnn import GCNN
import numpy as np
import scipy.sparse
import graph
import time
import os
mnist = input_data.read_data_sets('../../data/mnist', one_hot=False)

train_data = tf.data.Dataset.from_tensor_slices(
    mnist.train.images.astype(np.float32))
val_data = tf.data.Dataset.from_tensor_slices(
    mnist.validation.images.astype(np.float32))
test_data = tf.data.Dataset.from_tensor_slices(
    mnist.test.images.astype(np.float32))
train_labels = tf.data.Dataset.from_tensor_slices(
    mnist.train.labels)
val_labels = tf.data.Dataset.from_tensor_slices(
    mnist.validation.labels)
test_labels = tf.data.Dataset.from_tensor_slices(
    mnist.test.labels)

tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.app.flags
FLAGS = flags.FLAGS

# Graphs.
flags.DEFINE_integer('number_edges', 8, 'Graph: minimum number of edges per vertex.')
flags.DEFINE_string('metric', 'euclidean', 'Graph: similarity measure (between features).')
# TODO: change cgcnn for combinatorial Laplacians.
flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
flags.DEFINE_integer('coarsening_levels', 4, 'Number of coarsened graphs.')

# Directories.
flags.DEFINE_string('dir_data', os.path.join('..', 'data', 'mnist'), 'Directory to store data.')


def grid_graph(m, corners=False):
    z = graph.grid(m)
    dist, idx = graph.distance_sklearn_metrics(z, k=FLAGS.number_edges, metric=FLAGS.metric)
    A = graph.adjacency(dist, idx)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz//2, FLAGS.number_edges*m**2//2))
    return A

t_start = time.process_time()
A = grid_graph(28, corners=False)
A = graph.replace_random_edges(A, 0)
L = graph.laplacian(A, normalized=True)
print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
# graph.plot_spectrum(L)
del A

gcnn = GCNN(L, 10, 10, 10)
batch_size = 200
epochs = 20
prefetch_size = 1

train_data_copy = train_data.map(lambda x: tf.expand_dims(x, 1)).batch(batch_size).prefetch(prefetch_size)
train_labels_copy = train_labels.batch(batch_size).prefetch(prefetch_size)
train_data = train_data.map(lambda x: tf.expand_dims(x, 1)).batch(batch_size).prefetch(prefetch_size)
train_labels = train_labels.batch(batch_size).prefetch(prefetch_size)
val_data = val_data.map(lambda x: tf.expand_dims(x, 1)).batch(batch_size).prefetch(prefetch_size)
val_labels = val_labels.batch(batch_size).prefetch(prefetch_size)
test_data = test_data.map(lambda x: tf.expand_dims(x, 1)).batch(batch_size).prefetch(prefetch_size)
test_labels = test_labels.batch(batch_size).prefetch(prefetch_size)

data_format = (train_data.output_types, train_data.output_shapes)
labels_format = (train_labels.output_types, train_labels.output_shapes)
gcnn.build_graph(data_format, labels_format, 55000/batch_size, 0.95, 1, tf.get_default_graph())
sess = gcnn.fit(train_data, train_labels, val_data, val_labels,
                train_total=55000, val_total=5000, batch_size=batch_size,
                num_epochs=epochs, progress_per=100, eval_every_n_epochs=1)
print(gcnn.evaluate(test_data, test_labels, total=10000, sess=sess))
print(gcnn.evaluate(train_data_copy, train_labels_copy, total=55000, sess=sess))
