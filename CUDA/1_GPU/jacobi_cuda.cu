#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include <cuda.h>
#include <math.h>

#define sumBlockSize 1024
#define sumBlocksNum 40
__global__ void initialise_fXY(double *f,int n,double offset,double delta){
    int index = blockIdx.x * blockDim.x + threadIdx.x ;
    //if(index < n){
        f[index] = offset + index * delta;
    //}
}

__global__ void jacobiIterations(double *src,double *dest,
                                 double cx, double cy, double cc,
                                 double *fX,double *fY,
                                 double relax,double alpha,
                                 int dimX,int dimY,
                                 double *error){
    #define F(XX,YY) ( -alpha*(1.0-fX[XX]*fX[XX])*(1.0-fY[YY]*fY[YY]) - 2.0*(1.0-fX[XX]*fX[XX]) - 2.0*(1.0-fY[YY]*fY[YY]) )
    #define SRC(XX,YY) src[(YY)*dimX+(XX)]
    #define DST(XX,YY) dest[(YY)*dimX+(XX)]
    #define ERR(XX,YY) error[(YY)*dimX + (XX)]

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double updateVal;


    if((i < 1 || i > dimX - 2) || (j < 1 || j > dimY - 2)){
        return;
    }

    updateVal = (   (SRC(i-1,j) + SRC(i+1,j))*cx +
                                (SRC(i,j-1) + SRC(i,j+1))*cy +
                                SRC(i,j)*cc - F(i-1, j-1)
                            )/cc;
    DST(i,j) = SRC(i,j) - relax*updateVal;

    ERR(i,j) = updateVal * updateVal;
}

__global__ void sum(const double *in, int arraySize, double *out) {
    int start = threadIdx.x + blockIdx.x*blockDim.x;
    const int gridSize = sumBlockSize*gridDim.x;
    double sum = 0;
    for (int i = start; i < arraySize; i += gridSize)
        sum += in[i];
    __shared__ double shArr[sumBlockSize];
    shArr[threadIdx.x] = sum;
    __syncthreads();
    for (int size = sumBlockSize/2; size>0; size/=2) {
        if (threadIdx.x<size)
            shArr[threadIdx.x] += shArr[threadIdx.x+size];
        __syncthreads();
    }
    if (threadIdx.x == 0){
        out[blockIdx.x] = shArr[0];
    }
}

int main(int argc,char *argv[]){
	int n, m, mits, iterationCount = 0, x, y;
    double alpha, tol, relax, error = HUGE_VAL, *tmp, error_in = 0.0;

    // Read the Input Arguments
    scanf("%d,%d", &n, &m); scanf("%lf", &alpha);  scanf("%lf", &relax); scanf("%lf", &tol); scanf("%d", &mits);
    //printHeader(0, n, m, alpha, relax, tol, mits);

    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;

    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight-xLeft)/(n-1);

    double deltaY = (yUp-yBottom)/(m-1);

    // Coefficients
    double cx = 1.0/(deltaX*deltaX);
    double cy = 1.0/(deltaY*deltaY);
    double cc = -2.0*cx-2.0*cy-alpha;

    // Allocate the main matrices (Those two calls also zero the boundary elements)
    double *cu; cudaMalloc(&cu,(n+2)*(m+2)*sizeof(double));
    cudaMemset(cu,0,(n+2)*(m+2)*sizeof(double));
    double *cu_old; cudaMalloc(&cu_old,(n+2)*(m+2)*sizeof(double));
    cudaMemset(cu_old,0,(n+2)*(m+2)*sizeof(double));


    clock_t start = clock();
    // Allocate the precomputation buffers
    double *fX = (double *)malloc(n*sizeof(double));
    double *cfX; cudaMalloc(&cfX,n*sizeof(double));
    initialise_fXY<<<n,1>>>(cfX,n,xLeft,deltaX);
    cudaMemcpy(fX,cfX,n*sizeof(double),cudaMemcpyDeviceToHost);

    double *fY = (double *)malloc(n*sizeof(double));
    double *cfY; cudaMalloc(&cfY,n*sizeof(double));
    initialise_fXY<<<m,1>>>(cfY,m,yBottom,deltaY);
    cudaMemcpy(fY,cfY,n*sizeof(double),cudaMemcpyDeviceToHost);

    int numThreads = 32;
    dim3 blockSize(numThreads,numThreads);
    dim3 gridSize((m + numThreads - 1)/numThreads,(n + numThreads -1)/numThreads);

    double *cerror_in;cudaMalloc(&cerror_in,(m+2)*(n+2)*sizeof(double));
    cudaMemset(cerror_in,0,(m+2)*(n+2)*sizeof(double));

    double *csum;
    cudaMalloc(&csum,sumBlocksNum*sizeof(double));


    while(iterationCount < mits && error > tol){

        jacobiIterations<<<gridSize,blockSize>>>(cu_old,cu,cx,cy,cc,cfX,cfY,relax,alpha,n+2,m+2,cerror_in);
        if(cudaGetLastError() != cudaSuccess){
            printf("Error Launching Kernel in jacobiIterations\n");
            return 1;
        }
        cudaDeviceSynchronize();

        sum<<<sumBlocksNum, sumBlockSize>>>(cerror_in,(m+2)*(n+2), csum);       //partial sum of every block
        sum<<<1, sumBlockSize>>>(csum, sumBlocksNum, csum);     //final sum
        cudaDeviceSynchronize();

        cudaMemcpy(&error_in, csum, sizeof(double), cudaMemcpyDeviceToHost);
        error = sqrt(error_in)/(n*m); 

        tmp = cu_old;       //swap
        cu_old = cu;
        cu = tmp;

        iterationCount++;
    }

    double *u= (double*)malloc((n+2)*(m+2)*sizeof(double));
    cudaMemcpy(u,cu_old,(m+2)*(n+2)*sizeof(double),cudaMemcpyDeviceToHost);
    // Print system time
    int msec = (clock() - start) * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

    // Print the residual
    printf("Residual %g\n",error);

    // checkSolution() - calculate the error of the found solution

    error = 0;
    for (y = 1; y < (m+1); y++) {
        for (x = 1; x < (n+1); x++) {

            #define U(XX,YY) u[(YY)*(n+2)+(XX)]
            error_in = U(x,y) - (1.0-fX[x-1]*fX[x-1])*(1.0-fY[y-1]*fY[y-1]);
            error += error_in*error_in;
        }
    }
    printf("The error of the iterative solution is %g\n", sqrt(error)/(n*m));


    free(u);
    free(fX);free(fY);
    cudaFree(cu);cudaFree(cu_old);
    cudaFree(cfX);cudaFree(cfY);
    cudaFree(cerror_in);cudaFree(csum);

    return 0;
}