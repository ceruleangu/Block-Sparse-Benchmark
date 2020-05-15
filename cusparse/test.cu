#include <cusparse.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <unordered_map>
#include <map>
#include <set>

inline const char* GetCuSparseErrorString(int error) {
    switch (error) {
    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE:
      return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSPARSE_STATUS_ZERO_PIVOT:
      return "CUSPARSE_STATUS_ZERO_PIVOT";
    }
    return "Unrecognized error";
  }
  
#ifndef CHECK_CUSPARSE_ERROR
#define CHECK_CUSPARSE_ERROR(fn)                  \
do {                                          \
    int error = static_cast<int>(fn);                      \
    if (error != CUSPARSE_STATUS_SUCCESS) \
      std::cout << __LINE__ << " CUSPARSE: " << error << " " << GetCuSparseErrorString(error) << std::endl;\
} while (0)  // ; intentionally left off.
#endif  // CHECK_CUSPARSE_ERROR
  

struct GpuTimer
{
      cudaEvent_t start;
      cudaEvent_t stop;

      GpuTimer()
      {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
      }

      ~GpuTimer()
      {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
      }

      void Start()
      {
            cudaEventRecord(start, 0);
      }

      void Stop()
      {
            cudaEventRecord(stop, 0);
      }

      float Elapsed()
      {
            float elapsed;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            return elapsed;
      }
};


float get_random()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
    return dis(e);
}

int get_random_int(int max) {
    static std::default_random_engine e;
    std::uniform_int_distribution<int> dis(0, max);
    return dis(e);
}

void generate_candidate_blocks(int R, int C, int BS_R, int BS_C, int num_blocks, int *weight_indptr, int *weight_indices) {
    std::map<int, std::set<int>> blocks;
    int num_r_block = R / BS_R;
    int num_c_block = C / BS_C;
    int curr_size = 0;
    while (curr_size < num_blocks) {
        int r = get_random_int(num_r_block);
        int c = get_random_int(num_c_block);
        if (blocks[r].count(c) == 0) {
          blocks[r].insert(c);
          curr_size++;
        }
    }

    int current_ptr = 0;
    int i;
    for (i = 0; i < num_r_block; i++) {
        weight_indptr[i] = current_ptr;
        for (auto block : blocks[i]) {
          weight_indices[current_ptr++] = block;  
        }
    }
    weight_indptr[i] = current_ptr;
}
cusparseHandle_t handle{nullptr};

void test(int M, int N, int K, int BS_R, int BS_C, float density) {

    float *data;
    float *weight;
    int *weight_ind;
    int *weight_ptr;
    int nnz = int(density * M * N);
    int num_blocks = int(nnz / (BS_R * BS_C)) + 1;

    data = (float*)malloc(M*K*sizeof(float));
    weight = (float*)malloc(num_blocks*BS_R*BS_C*sizeof(float));
    weight_ind = (int*)malloc(num_blocks*sizeof(int));
    weight_ptr = (int*)malloc((N+1)*sizeof(int));

    for (int i = 0; i < M*K; i++) {
        data[i] = get_random();
    }

    for (int i = 0; i < num_blocks*BS_R*BS_C; i++) {
        weight[i] = get_random();
    }

    generate_candidate_blocks(N, K, BS_R, BS_C, num_blocks, weight_ptr, weight_ind);


    float *data_cuda;
    float *weight_cuda;
    int *weight_ind_cuda;
    int *weight_ptr_cuda;
    float *output_cuda;
    
    cudaMalloc(&data_cuda, M*K*sizeof(float));
    cudaMalloc(&weight_cuda, K*N*sizeof(float));
    cudaMalloc(&weight_ind_cuda, num_blocks*sizeof(int));
    cudaMalloc(&weight_ptr_cuda, (N+1)*sizeof(int));
    cudaMalloc(&output_cuda, M*N*sizeof(float));


    cudaMemcpy(data_cuda, data, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_cuda, weight, num_blocks*BS_R*BS_C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_ind_cuda, weight_ind, num_blocks*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_ptr_cuda, weight_ptr, (N+1)*sizeof(int), cudaMemcpyHostToDevice);

    cusparseMatDescr_t descr{nullptr};

    CHECK_CUSPARSE_ERROR(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE_ERROR(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE_ERROR(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    // A: sparse NxK
    // B: dense (transpose)MxK
    float alpha = 1.0;
    float beta = 0.0;

    GpuTimer timer;
    int n_runs = 20;
    timer.Start();
    for (int i = 0; i < n_runs; i++) {
    CHECK_CUSPARSE_ERROR(
      cusparseSbsrmm(handle,
               CUSPARSE_DIRECTION_ROW,
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               N/BS_R, M, K/BS_C, num_blocks, &alpha,
               descr, weight_cuda, weight_ptr_cuda, weight_ind_cuda, BS_R,
               data_cuda, K,
               &beta, output_cuda, N));
    }
    timer.Stop();
    std::cout << "M = " << M << " N = " << N << " K = " << K << " BS_R " << BS_R << " BS_C " << BS_C << " density " << density << std::endl;
    std::cout << timer.Elapsed() / n_runs << " ms" << std::endl;
}


int main() {
  CHECK_CUSPARSE_ERROR(cusparseCreate(&handle));
  for (int M : {1, 8, 64})
  for (int N : {1024}) 
  for (int K : {1024}) 
  for (int BS_R : {8, 16, 32}) 
  for (float density : {0.20, 0.15, 0.05}) 
  test(M, N, K, BS_R, BS_R, density);

  return 0;
}

