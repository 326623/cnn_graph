from numpy.polynomial import Polynomial
from numpy.polynomial.chebyshev import Chebyshev
import scipy.sparse
import tensorflow as tf
from joy_models import BaseModel
import graph
import numpy as np

# return a list of lists [coef of power 0, coef of power 1, ..., coef of power k-1]
# for the chebyshev polynomial.
# len(weight_power) == K
def chebyshev_list(K):
    coef_list = []
    for k in range(K):
        coef = Polynomial.cast(Chebyshev.basis(k))
        coef_list.append(coef.coef)
    return coef_list

class GCNN(BaseModel):
    def __init__(self, L, F, K, num_classes):
        super().__init__()
        # number of polynomial, hops: K,
        # number of filters: F,
        # Normalized Laplacian,
        self.L = L
        self.F = F
        self.K = K
        self.NCls = num_classes
        self.learning_rate = 0.02
        self.dropout = 1
        self.regularization = 5e-4

    def chebyshev_p(self, x, L, Fout, K):
        # Fout: num of output features
        # N: number of signal, batch size
        # V: number of vertices, graph size
        # Fin: number of features per signal
        N, V, Fin = x.get_shape()
        L = scipy.sparse.csr_matrix(L)
        # convert to a list of chebyshev matrix
        base_L = graph.rescale_L(L, lmax=2)
        coef_list = chebyshev_list(K)
        chebyshev_Ls = []
        for coef in coef_list:
            L = 0
            for i in range(len(coef)):
                L += coef[i] * (base_L**i)
            chebyshev_Ls.append(L)

        # convert to sparseTensor
        def convert2Sparse(L):
            L = L.tocoo()
            indices = np.column_stack((L.row, L.col))
            print(len(indices))
            L = tf.SparseTensor(indices, L.data, L.shape)
            return tf.sparse_reorder(L)
        chebyshev_Ls = map(lambda L: convert2Sparse(L), chebyshev_Ls)

        # chebyshev filtering
        # N x V x Fin -> N x Fin x V -> Fin*N x V -> V x Fin*N
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.reshape(x, [-1, V])
        x = tf.transpose(x)

        x_filtered = []
        for T in chebyshev_Ls:
            # T: V x V, x: V x Fin*N, output: V x Fin*N
            x_filtered.append(tf.sparse_tensor_dense_matmul(T, x))
            # T: V x V, x: N x V x Fin, output: N x V x Fin
            # x_filtered.append(tf.map_fn(lambda x: tf.sparse_tensor_dense_matmul(T, x), x))

        # K x N x V x Fin
        # x = tf.stack(x_filtered)
        # x = tf.parallel_stack(x_filtered)

        # K x V x Fin*N -> K x V x Fin x N -> N x V x Fin x K
        x = tf.stack(x_filtered)
        x = tf.reshape(x, [K, V, Fin, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])

        # K x N x V x Fin -> N x V x Fin x K
        # x = tf.transpose(x, perm=[1, 2, 3, 0])
        x = tf.reshape(x, [-1, Fin*K])
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W) # N*V x Fout
        x = tf.nn.relu(x)
        return tf.reshape(x, [-1, V, Fout])

    def chebyshev(self, x, L, Fout, K):
        # Fout: num of output features
        # N: number of signal, batch size
        # V: number of vertices, graph size
        # Fin: number of features per signal
        N, V, Fin = x.get_shape()
        L = scipy.sparse.csr_matrix(L)
        # convert to a list of chebyshev matrix
        L = graph.rescale_L(L, lmax=2)

        # convert to sparseTensor
        def convert2Sparse(L):
            L = L.tocoo()
            indices = np.column_stack((L.row, L.col))
            print(len(indices))
            L = tf.SparseTensor(indices, L.data, L.shape)
            return tf.sparse_reorder(L)
        L = convert2Sparse(L)

        # chebyshev filtering
        # N x V x Fin -> V x Fin x N -> V x Fin*N
        x = tf.transpose(x, perm=[1, 2, 0])
        x = tf.reshape(x, [V, -1])
        x0 = x
        x = tf.expand_dims(x, 0)

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x V x Fin*N
            return tf.concat([x, x_], axis=0) # K x V x Fin*N

        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)

        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0
            x = concat(x, x2)
            x0, x1 = x1, x2

        # K x V x Fin*N -> K x V x Fin x N -> N x V x Fin x K -> N*V x Fin*K
        x = tf.reshape(x, [K, V, Fin, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])
        x = tf.reshape(x, [-1, Fin*K])

        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W) # N*V x Fout
        return tf.reshape(x, [-1, V, Fout])

    def chebyshev5(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, -1])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [-1, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [-1, M, Fout])  # N x M x Fout

    def fc(self, x, Mout, relu=True):
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _inference(self, x, dropout):
        N, V, Fin = x.get_shape()
        Fout = self.F # number of filters
        K = self.K # support size of filter
        with tf.variable_scope('gconv1'):
            x = self.chebyshev(x, self.L, Fout, K)
            x = tf.nn.relu(x)

        # with tf.variable_scope('gconv2'):
        #     x = self.chebyshev(x, L, Fout, K)
        #     x = tf.nn.dropout(x, dropout)

        with tf.variable_scope('logits'):
            x = tf.reshape(x, [-1, V * Fout])
            y = self.fc(x, self.NCls, relu=False)
        return y
