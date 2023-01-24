

all: jacobi_cuda

jacobi_cuda:
	nvcc -o jacobi_cuda jacobi_cuda2gpus2.cu -Xcompiler " -fopenmp"

run: all
	qsub ./CudaPBSScript.sh
