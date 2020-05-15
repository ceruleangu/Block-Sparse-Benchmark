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


"""
Benchmark for sparse operator [Gray et al., 2017] 
on block sparse matrices.
https://s3-us-west-2.amazonaws.com/openai-assets/blocksparse/blocksparsepaper.pdf
"""
import numpy as np
import scipy.sparse as sp
from blocksparse.matmul import BlocksparseMatMul
import tensorflow as tf
import itertools


def random_bsr_matrix_helper(M, N, BS_R, BS_C, density, dtype):
    """Generate a random BSR matrix, return the sparse pattern"""
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


def test_sparse_dense_bsr_gray(minibatch_size, N, K, BS_R, BS_C, density):    
    """Run and profile BSR dense with tensorflow"""
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

    # Run
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


    run_metadata = tf.RunMetadata()
    sess.run([y], run_metadata=run_metadata, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))

    # Print to stdout an analysis of the memory usage and the timing information
    # broken down by python codes.
    ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
    
    # Print to stdout an analysis of the memory usage and the timing information
    # broken down by operation types.
    tf.profiler.profile(
        tf.get_default_graph(),
        run_meta=run_metadata,
        cmd='op',
        options=tf.profiler.ProfileOptionBuilder.time_and_memory())

    tf.reset_default_graph()


if __name__ == "__main__":
    for M in [8]:
        for N in [1024]:
            for K in [128]:
                for BS_R in [16]:
                    BS_C = BS_R
                    for density in [0.2, .15, 0.05]:
                        test_sparse_dense_bsr_gray(M, N, K, BS_R, BS_C, density)
