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


def _sparse_dense_bsrmm(data, weight_data, weight_indices, weight_indptr):
    (m, k) = get_const_tuple(data.shape)
    (_, bs_r, bs_c) = get_const_tuple(weight_data.shape)
    (num_blocks_plus_1, ) = get_const_tuple(weight_indptr.shape)
    num_blocks = num_blocks_plus_1 - 1
    #cached_data = te.compute(data.shape, lambda *idx: data(*idx), name='T_cache_data')

    def _compute_block(i, nb_j, j):
        row_start = weight_indptr[nb_j]
        row_end = weight_indptr[nb_j + 1]
        row_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, k // bs_c), name="elem_idx")
        # elem_idx = te.reduce_axis(
            # (0, row_elems), name="elem_idx")
        block_offset = row_start + elem_idx
        c = te.reduce_axis((0, bs_c), name="c")
        block_j = weight_indices[block_offset]
        block_ij_val = weight_data[block_offset][j][c]
        x_val = data[i, bs_c * block_j + c]
        prod = block_ij_val * x_val
        return te.sum(te.if_then_else(elem_idx < row_elems, prod, tvm.tir.const(0.0, dtype=data.dtype)), axis=[elem_idx, c])

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    bsrmm_block = te.compute(
        (m, num_blocks, bs_r), _compute_block,
        tag="sparse_dense_bsrmm_block", name="T_bsrmm_block")
    return te.compute(
        (m, num_blocks * bs_r),
        lambda m, n: bsrmm_block[m, idxd(n, bs_r), idxm(n, bs_r)],
        tag="sparse_dense_bsrmm", name="T_bsrmm")





# baseline
def schedule_sparse_dense_cuda_baseline(outs):
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



def schedule_sparse_dense_cuda_per_tile(outs):
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
            

            (m_o, n_o) = s[y_reshape].op.axis
            (noo, noi) = s[y_reshape].split(n_o, bs_r)
            s[y_reshape].bind(m_o, te.thread_axis("blockIdx.x"))
            s[y_reshape].bind(noo, te.thread_axis("threadIdx.x"))
            s[y_bsrmm].compute_at(s[y_reshape], noo)

            # s[y_bsrmm].reorder(num_blocks, m, elem_idx, b_r, c)
            # s[y_bsrmm].vectorize(b_r)

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




def schedule_sparse_dense_cuda_simple_local_cache(outs):
    """Create schedule for sparse dense"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "sparse_dense_bsrmm":
            y_bsrmm = op.input_tensors[0]
            print(y_bsrmm.op.input_tensors)
            data = y_bsrmm.op.input_tensors[3]
            #data_original = data.op.input_tensors[0]
            assert y_bsrmm.op.tag == "sparse_dense_bsrmm_block"
            y_reshape = op
            (m, num_blocks, b_r) = s[y_bsrmm].op.axis
            bs_r = get_const_int(b_r.dom.extent)
            (elem_idx, c) = s[y_bsrmm].op.reduce_axis
            
            # cache
            # data_shared = s.cache_read(data, 'shared', [y_bsrmm])
            data_local = s.cache_read(data, 'shared', [y_bsrmm])
            s[data_local].vectorize(s[data_local].split(s[data_local].op.axis[1], 4)[1])
            s[data_local].compute_at(s[y_bsrmm], b_r)
            (m_o, n_o) = s[y_reshape].op.axis
            s[y_reshape].bind(m_o, te.thread_axis("blockIdx.x"))
            s[y_reshape].bind(n_o, te.thread_axis("blockIdx.y"))
            s[y_bsrmm].compute_at(s[y_reshape], s[y_reshape].op.axis[1])

            # vectorized copy
            # noi_o, noi_i = s[y_reshape].split(noi, 4)
            # s[y_reshape].vectorize(noi_i)


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


def schedule_sparse_dense_cuda_optimize(outs):
    """Create schedule for sparse dense"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "sparse_dense_bsrmm":
            y_bsrmm = op.input_tensors[0]
            print(y_bsrmm.op.input_tensors)
            data = y_bsrmm.op.input_tensors[3]
            assert y_bsrmm.op.tag == "sparse_dense_bsrmm_block"
            y_reshape = op
            (m, num_blocks, b_r) = s[y_bsrmm].op.axis
            bs_r = get_const_int(b_r.dom.extent)
            (elem_idx, c) = s[y_bsrmm].op.reduce_axis
            
            # cache
            # data_shared = s.cache_read(data, 'shared', [y_bsrmm])

            # split 
            (m_o, n_o) = s[y_reshape].op.axis
            (noo, noi) = s[y_reshape].split(n_o, bs_r)

            s[y_bsrmm].reorder(num_blocks, m, elem_idx, b_r, c)
            # s[data_shared].compute_at(s[y_bsrmm], b_r)
            s[y_reshape].bind(m_o, te.thread_axis("blockIdx.x"))
            s[y_reshape].bind(n_o, te.thread_axis("blockIdx.y"))
            s[y_bsrmm].compute_at(s[y_reshape], n_o)

            thread_x = te.thread_axis("threadIdx.x")
            co, ci = s[y_bsrmm].split(c, 8)
            y_bsrmm_factored = s.rfactor(y_bsrmm, ci)
            tx = s[y_bsrmm].op.reduce_axis[0]
            s[y_bsrmm].bind(tx, thread_x)
            s[y_bsrmm_factored].compute_at(s[y_bsrmm], tx)
            s[y_bsrmm].set_store_predicate(thread_x.var.equal(0))


            # s[y_bsrmm].bind(c, thread_x)

            # vectorized copy
            # noi_o, noi_i = s[y_reshape].split(noi, 4)
            # s[y_reshape].vectorize(noi_i)


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


def schedule_sparse_dense_cuda_allreduce_cache_w_indptr(outs):
    """Create schedule for sparse dense"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "sparse_dense_bsrmm":
            y_bsrmm = op.input_tensors[0]
            w_indptr = y_bsrmm.op.input_tensors[0]
            print(y_bsrmm.op.input_tensors)
            assert y_bsrmm.op.tag == "sparse_dense_bsrmm_block"
            y_reshape = op
            w_indptr_local = s.cache_read(w_indptr, 'local', [y_bsrmm])
            (m, num_blocks, b_r) = s[y_bsrmm].op.axis
            bs_r = get_const_int(b_r.dom.extent)
            (elem_idx, c) = s[y_bsrmm].op.reduce_axis
            
            # cache
            # data_shared = s.cache_read(data, 'shared', [y_bsrmm])

            # split 
            (m_o, n_o) = s[y_reshape].op.axis
            # (noo, noi) = s[y_reshape].split(n_o, bs_r)

            # s[y_bsrmm].reorder(num_blocks, m, elem_idx, b_r, c)
            # s[data_shared].compute_at(s[y_bsrmm], b_r)
            s[y_reshape].bind(m_o, te.thread_axis("blockIdx.x"))
            s[y_reshape].bind(n_o, te.thread_axis("blockIdx.y"))
            s[y_bsrmm].compute_at(s[y_reshape], n_o)

            thread_x = te.thread_axis("threadIdx.x")
            co, ci = s[y_bsrmm].split(c, 8)
            y_bsrmm_factored = s.rfactor(y_bsrmm, ci)
            s[w_indptr_local].compute_at(s[y_bsrmm_factored], s[y_bsrmm_factored].op.axis[-1])
            tx = s[y_bsrmm].op.reduce_axis[0]
            s[y_bsrmm].bind(tx, thread_x)
            s[y_bsrmm_factored].compute_at(s[y_bsrmm], tx)
            s[y_bsrmm].set_store_predicate(thread_x.var.equal(0))
            s[y_reshape].set_store_predicate(thread_x.var.equal(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def schedule_sparse_dense_cuda_allreduce_cache_w_indptr_w_indices(outs):
    """Create schedule for sparse dense"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "sparse_dense_bsrmm":
            y_bsrmm = op.input_tensors[0]
            w_indptr = y_bsrmm.op.input_tensors[0]
            w_indices = y_bsrmm.op.input_tensors[2]
            w_data = y_bsrmm.op.input_tensors[1]
            print(y_bsrmm.op.input_tensors)
            assert y_bsrmm.op.tag == "sparse_dense_bsrmm_block"
            y_reshape = op
            (m, num_blocks, b_r) = s[y_bsrmm].op.axis
            bs_r = get_const_int(b_r.dom.extent)
            (elem_idx, c) = s[y_bsrmm].op.reduce_axis
            w_indptr_local = s.cache_read(w_indptr, 'local', [y_bsrmm])
            w_indices_local = s.cache_read(w_indices, 'shared', [y_bsrmm])
            w_data_local = s.cache_read(w_data, 'local', [y_bsrmm])
            
            (m_o, n_o) = s[y_reshape].op.axis
            s[y_reshape].bind(m_o, te.thread_axis("blockIdx.x"))
            s[y_reshape].bind(n_o, te.thread_axis("blockIdx.y"))
            s[y_bsrmm].compute_at(s[y_reshape], n_o)

            thread_x = te.thread_axis("threadIdx.x")
            co, ci = s[y_bsrmm].split(c, 8)
            y_bsrmm_factored = s.rfactor(y_bsrmm, ci)

            s[w_indptr_local].compute_at(s[y_bsrmm_factored], s[y_bsrmm_factored].op.axis[-1])
            s[w_indices_local].compute_at(s[y_bsrmm_factored], s[y_bsrmm_factored].op.axis[-1])
            s[w_data_local].compute_at(s[y_bsrmm_factored], s[y_bsrmm_factored].op.axis[-1])
            _, tx = s[w_indices_local].split(s[w_indices_local].op.axis[0], 8)
            s[w_indices_local].bind(tx, thread_x)

            tx = s[y_bsrmm].op.reduce_axis[0]
            s[y_bsrmm].bind(tx, thread_x)
            s[y_bsrmm_factored].compute_at(s[y_bsrmm], tx)
            s[y_bsrmm].set_store_predicate(thread_x.var.equal(0))
            s[y_reshape].set_store_predicate(thread_x.var.equal(0))

    traverse_inline(s, outs[0].op, _callback)
    return s
def schedule_sparse_dense_cuda_allreduce(outs):
    """Create schedule for sparse dense"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "sparse_dense_bsrmm":
            y_bsrmm = op.input_tensors[0]
            w_indptr = y_bsrmm.op.input_tensors[0]
            print(y_bsrmm.op.input_tensors)
            assert y_bsrmm.op.tag == "sparse_dense_bsrmm_block"
            y_reshape = op
            (m, num_blocks, b_r) = s[y_bsrmm].op.axis
            bs_r = get_const_int(b_r.dom.extent)
            (elem_idx, c) = s[y_bsrmm].op.reduce_axis
            
            (m_o, n_o) = s[y_reshape].op.axis
            s[y_reshape].bind(m_o, te.thread_axis("blockIdx.x"))
            s[y_reshape].bind(n_o, te.thread_axis("blockIdx.y"))
            s[y_bsrmm].compute_at(s[y_reshape], n_o)

            thread_x = te.thread_axis("threadIdx.x")
            co, ci = s[y_bsrmm].split(c, 8)
            y_bsrmm_factored = s.rfactor(y_bsrmm, ci)
            tx = s[y_bsrmm].op.reduce_axis[0]
            s[y_bsrmm].bind(tx, thread_x)
            s[y_bsrmm_factored].compute_at(s[y_bsrmm], tx)
            s[y_bsrmm].set_store_predicate(thread_x.var.equal(0))
            s[y_reshape].set_store_predicate(thread_x.var.equal(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


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
    # Y = _sparse_dense_bsrmm(X, W_data, W_indices, W_indptr)
    # s = te.create_schedule(Y.op)
    #s = schedule_sparse_dense_cuda_baseline([Y])
    # s = schedule_sparse_dense_cuda_per_tile([Y])

    # s = schedule_sparse_dense_cuda_optimize([Y])
    s = schedule_sparse_dense_cuda_allreduce([Y])
    # s = schedule_sparse_dense_cuda_allreduce_cache_w_indptr_w_indices([Y])
    # s = schedule_sparse_dense_cuda_simple_local_cache([Y])
    
    # print(tvm.lower(s, [X, W_data, W_indices, W_indptr, Y], simple_mode=True))
    func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
    # print('CUDA')
    # print(func.imported_modules[0].get_source())
    # print('----------------------------')
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
        for N, K in [(1024,1024)]:
        #for N, K in [(128,768),(1024,1024)]:
            for M in [1, 8]:
                for BS_R in [8, 16, 32]:
                    BS_C = BS_R
                    for density in [0.2, .15, 0.05]:
                        test_sparse_dense_bsr(M, N, K, BS_R, BS_C, density)
                    print('=========')