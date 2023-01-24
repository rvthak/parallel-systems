#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include <cuda.h>
#include <math.h>
#include <omp.h>

#define sumBlockSize 1024
#define sumBlocksNum 24
__global__ void initialise_fXY(double *f,int n,double offset,double delta){
    int index = blockIdx.x * blockDim.x + threadIdx.x ;
    //if(index < n){
        f[index] = offset + index * delta;
    //}
}

//jacobiIterations<<<gridSize,blockSize>>>(cu_old,cu,cx,cy,cc,cf,relax,n+2,m+2,cerror_in);
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
    int n, m, mits, x, y;
    int iterationCount[2];
    iterationCount[0] = 0;
    iterationCount[1] = 0;
    double alpha, tol, relax;
    double error[2];
    error[0] = HUGE_VAL; error[1] = HUGE_VAL;
    double error_in[2];
    error_in[0] = 0.0;error_in[1] = 0.0;
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

    double *fX = (double *)malloc(n*sizeof(double));
    double *ctfX; cudaMalloc(&ctfX,n*sizeof(double));
    
    double *fY = (double *)malloc(n*sizeof(double));
    double *ctfY; cudaMalloc(&ctfY,n*sizeof(double));

    initialise_fXY<<<n,1>>>(ctfX,n,xLeft,deltaX);
    cudaMemcpy(fX,ctfX,n*sizeof(double),cudaMemcpyDeviceToHost);

    initialise_fXY<<<m,1>>>(ctfY,m,yBottom,deltaY);
    cudaMemcpy(fY,ctfY,m*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double *cu_old[2];
    double *cu[2];
    double *cerror_in[2];
    double *csum[2];
    double *cfX[2];
    double *cfY[2];
    double res[2];
    res[0]=0;res[1]=0;
    double *buf[2];
    buf[0] = (double *)malloc((n+2)*sizeof(double));
    buf[1] = (double *)malloc((n+2)*sizeof(double));


    double *u = (double *)malloc((m+2)*(n+2)*sizeof(double));

    int omp_numthreads = omp_get_max_threads();
    if(omp_numthreads==2){
        m = m/omp_numthreads;
    }else if(omp_numthreads!=1){
        return 1;
    }
    int numThreads = 32;

    dim3 blockSize(numThreads,numThreads);
    dim3 gridSize((m + numThreads - 1)/numThreads,(n + numThreads -1)/numThreads);

    double *tmp[2];

    cudaFree(ctfX);cudaFree(ctfY);

    clock_t start = clock();
    #pragma omp parallel
    {
        int ID =omp_get_thread_num();

        int TotalGpuNum;
        cudaGetDeviceCount(&TotalGpuNum );
        cudaSetDevice(ID % TotalGpuNum);

        cudaMalloc(&cfX[ID],n*sizeof(double));
        cudaMalloc(&cfY[ID],m*sizeof(double));
        
        cudaMemcpy(cfX[ID],fX,n*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(cfY[ID],fY+ID*m,m*sizeof(double),cudaMemcpyHostToDevice);
        
        cudaMalloc(&cerror_in[ID],(m+2)*(n+2)*sizeof(double));
        cudaMemset(cerror_in[ID],0,(m+2)*(n+2)*sizeof(double));

        cudaMalloc(&csum[ID],sumBlocksNum*sizeof(double));

        cudaMalloc(&cu[ID],(n+2)*(m+2)*sizeof(double));
        cudaMemset(cu[ID],0,(n+2)*(m+2)*sizeof(double));

        cudaMalloc(&cu_old[ID],(n+2)*(m+2)*sizeof(double));
        cudaMemset(cu_old[ID],0,(n+2)*(m+2)*sizeof(double));
        
        while(iterationCount[ID] < mits && error[ID] > tol){
            error_in[ID] =0;
            
            jacobiIterations<<<gridSize,blockSize>>>(cu_old[ID],cu[ID],cx,cy,cc,cfX[ID],cfY[ID],relax,alpha,n+2,m+2,cerror_in[ID]);
            if(cudaGetLastError() != cudaSuccess){
                printf("Error Launching Kernel in jacobiIterations %d\n",ID);
                exit(1);
            }
            cudaDeviceSynchronize();

            sum<<<sumBlocksNum, sumBlockSize>>>(cerror_in[ID],(m+2)*(n+2), csum[ID]); //partial sum
            sum<<<1, sumBlockSize>>>(csum[ID], sumBlocksNum, csum[ID]);  //the final sum
            cudaDeviceSynchronize();

            cudaMemcpy(res+ID, csum[ID], sizeof(double), cudaMemcpyDeviceToHost);
            
            
            error_in[ID]+=res[ID];            
            error[ID] = sqrt(error_in[ID])/(n*m); 
        

            tmp[ID] = cu_old[ID];
            cu_old[ID] = cu[ID];
            cu[ID] = tmp[ID];
            
            
            cudaMemcpy(buf[ID],cu_old[ID]+(m-ID*(m-1))*(n+2),(n+2)*sizeof(double),cudaMemcpyDeviceToHost);
            
            #pragma omp barrier
            
            cudaMemcpy(cu_old[ID]+(1-ID)*(m+1)*(n+2),buf[1-ID],(n+2)*sizeof(double),cudaMemcpyHostToDevice);
            
            
            iterationCount[ID]++;
        }

        cudaMemcpy(u+ID*(m+1)*(n+2),cu_old[ID],(m+1)*(n+2)*sizeof(double),cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        
        cudaFree(cu[ID]);cudaFree(cu_old[ID]);
        cudaFree(cfX[ID]);cudaFree(cfY[ID]);
        cudaFree(cerror_in[ID]);
        cudaFree(csum[ID]);

    }
    
    m = m*omp_numthreads;
    double ferror = sqrt(error_in[0] + error_in[1])/(n*m); 
    double ferror_in;
    
    int msec = (clock() - start) * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

    // Print the residual
    printf("Residual %g\n",ferror);

    // checkSolution() - calculate the error of the found solution

    ferror = 0;
    for (y = 1; y < (m+1); y++) {
        for (x = 1; x < (n+1); x++) {
            #define U(XX,YY) u[(YY)*(n+2)+(XX)]
            ferror_in = U(x,y) - (1.0-fX[x-1]*fX[x-1])*(1.0-fY[y-1]*fY[y-1]);
            ferror += ferror_in*ferror_in;
        }
    }
    printf("The error of the iterative solution is %g\n", sqrt(ferror)/(n*m));


    free(u);
    free(buf[0]); free(buf[1]);
    free(fX);free(fY);


    return 0;


}