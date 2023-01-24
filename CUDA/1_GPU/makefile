
all: jacobi_cuda

jacobi_cuda:
	nvcc -o jacobi_cuda jacobi_cuda.cu

run: all
	qsub ./CudaPBSScript.sh
