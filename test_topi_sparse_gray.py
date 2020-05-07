# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test code for sparse operator"""
import numpy as np
import tvm
from tvm import te
import topi
import topi.testing
from topi.util import get_const_tuple, traverse_inline, get_const_int
import tvm.contrib.sparse as tvmsp
from collections import namedtuple
import time
import scipy.sparse as sp


from blocksparse.matmul import BlocksparseMatMul
import tensorflow as tf
from tensorflow.python.profiler import model_analyzer

target = 'cuda'
context = tvm.context(target, 0)

def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype):
    import itertools
    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r:r + BS_R, c:c + BS_C] = np.random.randn(BS_R, BS_C)
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.indices.shape == (num_blocks, )
    assert s.indptr.shape == (M // BS_R + 1, )
    return s


def random_bsr_matrix_helper(M, N, BS_R, BS_C, density, dtype):
    import itertools
    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)]
    s = np.zeros((M//BS_R, N//BS_C), 'int32')
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        r //= BS_R
        c //= BS_C
        s[r, c] = 1

    return s


def schedule_sparse_dense_cuda(outs):
    """Create schedule for sparse dense"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "sparse_dense_bsrmm":
            y_bsrmm = op.input_tensors[0]
            assert y_bsrmm.op.tag == "sparse_dense_bsrmm_block"
            y_reshape = op
            (m, num_blocks, b_r) = s[y_bsrmm].op.axis
            bs_r = get_const_int(b_r.dom.extent)
            (elem_idx, c) = s[y_bsrmm].op.reduce_axis
            
            s[y_reshape].bind(s[y_reshape].op.axis[0], te.thread_axis("blockIdx.x"))
            s[y_reshape].bind(s[y_reshape].op.axis[1], te.thread_axis("threadIdx.x"))
            s[y_bsrmm].compute_at(s[y_reshape], s[y_reshape].op.axis[1])

            # s[y_bsrmm].reorder(num_blocks, m, elem_idx, b_r, c)
            # s[y_bsrmm].vectorize(b_r)
            # (m_o, n_o) = s[y_reshape].op.axis
            # (noo, noi) = s[y_reshape].split(n_o, bs_r)
            # s[y_bsrmm].compute_at(s[y_reshape], noi)
            # s[y_reshape].vectorize(noi)
            # if op != s[outs[0]].op:
            #     (y_o, y_i) = s[outs[0].op].split(
            #         s[outs[0].op].op.axis[1], 2 * simd_width)
            #     s[y_reshape].compute_at(s[outs[0]], y_o)
            #     s[outs[0].op].parallel(y_o)
            #     s[outs[0].op].vectorize(y_i)
            # else:
            #     m_o_noo = s[y_reshape].fuse(m_o, noo)
            #     s[y_reshape].parallel(m_o_noo)

    traverse_inline(s, outs[0].op, _callback)
    return s


def test_sparse_dense_bsr_gray(minibatch_size, N, K, BS_R, BS_C, density):    
    print("testing param", minibatch_size, N, K, BS_R, BS_C, density)
    # Initialize the sparse matrix multiplication object
    feature_axis = 0 if BS_R in [8, 16] else 1

    # Create a (random) sparsity pattern
    sparsity = random_bsr_matrix_helper(K, N, BS_R, BS_C, density, 'float32')
    bsmm = BlocksparseMatMul(sparsity, block_size=BS_R, feature_axis=feature_axis)
    # Initialize block-sparse weights
    w = tf.get_variable("w", bsmm.w_shape, dtype=tf.float32)

    if feature_axis==0:
        # Input to graph
        x = tf.get_variable("x", [K, minibatch_size], dtype=tf.float32)
    else:
        # Input to graph
        x = tf.get_variable("x", [minibatch_size, K], dtype=tf.float32)

    # Block-sparse matrix multiplication
    y = bsmm(x, w)
    # print(x.shape)
    # print(bsmm.w_shape)
    # print(y.shape)

    # Run
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


    run_metadata = tf.RunMetadata()
    sess.run([y], run_metadata=run_metadata, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))

    # Print to stdout an analysis of the memory usage and the timing information
    # broken down by python codes.
    ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
    opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()
        ).with_node_names(show_name_regexes=['.*my_code.py.*']).build()
    
    # tf.profiler.profile(
    #     tf.get_default_graph(),
    #     run_meta=run_metadata,
    #     cmd='code',
    #     options=opts)

    # Print to stdout an analysis of the memory usage and the timing information
    # broken down by operation types.
    tf.profiler.profile(
        tf.get_default_graph(),
        run_meta=run_metadata,
        cmd='op',
        options=tf.profiler.ProfileOptionBuilder.time_and_memory())

    tf.reset_default_graph()



def test_sparse_dense_bsr(M, N, K, BS_R, BS_C, density):
    # M, N, K, BS_R, BS_C, density = 1, 64, 128, 8, 16, 0.9
    print("testing param", M, N, K, BS_R, BS_C, density)
    X_np = np.random.randn(M, K).astype("float32")
    W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
    W_np = W_sp_np.todense()
    Y_np = X_np.dot(W_np.T)

    W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype), name='W_data')
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype), name='W_indices')
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype), name='W_indptr')
    X = te.placeholder(shape=X_np.shape, dtype=str(X_np.dtype), name='X')
    Y = topi.nn.sparse_dense(X, W_data, W_indices, W_indptr)
    # s = te.create_schedule(Y.op)
    s = schedule_sparse_dense_cuda([Y])
    
    print(tvm.lower(s, [X, W_data, W_indices, W_indptr, Y], simple_mode=True))
    func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
    timer = func.time_evaluator(func.entry_name, context, number=20)
    Y_tvm = tvm.nd.array(np.zeros(Y_np.shape, dtype=Y_np.dtype), ctx=context)
    mean_time = timer(tvm.nd.array(X_np, ctx=context),
                      tvm.nd.array(W_sp_np.data, ctx=context),
                      tvm.nd.array(W_sp_np.indices, ctx=context),
                      tvm.nd.array(W_sp_np.indptr, ctx=context),
                      Y_tvm).mean
    
    print('%g ms' % (mean_time * 1e3))
    print("------------------------")
    # tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np, atol=1e-4, rtol=1e-4)

if __name__ == "__main__":
    with tvm.target.create(target):
        for M in [8]:
            for N in [1024]:
                for K in [128]:
                    for BS_R in [16]:
                        BS_C = BS_R
                        for density in [0.2, .15, 0.05]:
                            test_sparse_dense_bsr_gray(M, N, K, BS_R, BS_C, density)
                            print('=========================================================================')
