# Benchmark for sparse-dense matrix multiplications

Benchmark for matrix multiplications between dense and block sparse (BSR) matrix in TVM, [blocksparse (Gray et al.)](https://github.com/openai/blocksparse) and cuSparse.

Please refer to [our paper](./Optimizing_Block_Sparse_Matrix_Multiplications_on_CUDA_with_TVM.pdf) for details.

Benchmark for TVM:
```
python3 test_topi_sparse.py --help
usage: test_topi_sparse.py [-h] [--n_trial N_TRIAL] [--repeat REPEAT] [--tune]
                           [--autotvm_log AUTOTVM_LOG]
                           {PEP,PTP,PROB,PRWB,PRWB_AT}
```
positional arguments: PEP, PTP, PROB, PRWB, PRWB_AT
```
PEP       Per Element Parallization  
PTP       Per Tile Paralliza-tion.  
PROB      Parallel Reduction Over Blocks. 
PRWB      Parallel Reduction With Blocks.  
PRWB_AT   Parallel Reduction Within Blocks + AutoTuning
```

optional arguments:
```
  -h, --help            show this help message and exit
  --n_trial N_TRIAL     Number of trials for AutoTVM
  --repeat REPEAT       Number of repeat for AutoTVM to profile on the device
  --tune                Enable AutoTVM tuning for setting PRWB_AT
  --autotvm_log AUTOTVM_LOG
                        Log file for auto tuning
```
 
Benchmark for blocksparse [Gray et al., 2017]:
```
python3 test_blocksparse_gray.py
```

Benchmark for cuSparse:
```
cd cusparse
make
./test
```