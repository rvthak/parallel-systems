
// Ioannis Rovithakis | sdi1800164
// Georgios Galanis   | sdi1800024

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// UI/Print functions
void printHeader(int mode, int n, int m, double alpha, double relax, double tol, int mits);
void printBuf(double *buf, int height, int width);

// mpicc jacobi_serial.c utils.c -o jacobi_serial.x -lm -Wall
// mpirun -np 1 ./jacobi_serial.x <input
int main(int argc, char **argv){

    int n, m, mits, iterationCount = 0, x, y;
    double alpha, tol, relax, error = HUGE_VAL, *tmp, error_in = 0.0, updateVal;

    // Read the Input Arguments
    x=scanf("%d,%d", &n, &m); x=scanf("%lf", &alpha);  x=scanf("%lf", &relax); x=scanf("%lf", &tol); x=scanf("%d", &mits);
    printHeader(0, n, m, alpha, relax, tol, mits);

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
    double *u     = (double*)calloc((n+2)*(m+2), sizeof(double)); //reverse order
    double *u_old = (double*)calloc((n+2)*(m+2), sizeof(double));
    if (u == NULL || u_old == NULL) { printf("Not enough memory for two %ix%i matrices\n", n+2, m+2); exit(1); }

    // Allocate the precomputation buffers
    double *fX = (double *)malloc(n*sizeof(double));
    double *fY = (double *)malloc(m*sizeof(double));
    if (fX == NULL || fY == NULL ) {  printf("Not enough memory for fX, fY matrices\n");  exit(1); }

    clock_t start = clock();

    // Pre-Compute fx, fy and f and store for future access - reuse
    for(y = 1; y < (m+1); y++ ){  fY[y-1] = yBottom + (y-1)*deltaY; }
    for(x = 1; x < (n+1); x++ ){  fX[x-1] = xLeft   + (x-1)*deltaX; }

    // Init MPI and Start timers before starting the calculations
    MPI_Init(NULL,NULL);
    double start_time;
    start_time = MPI_Wtime();

    /* Iterate as long as it takes to meet the convergence criterion */
    while (iterationCount < mits && error > tol) {  
      	
        #define SRC(XX,YY) u_old[(YY)*(n+2)+(XX)]
        #define DST(XX,YY) u[(YY)*(n+2)+(XX)]
        #define F(XX,YY) ( -alpha*(1.0-fX[XX]*fX[XX])*(1.0-fY[YY]*fY[YY]) - 2.0*(1.0-fX[XX]*fX[XX]) - 2.0*(1.0-fY[YY]*fY[YY]) )

        error_in = 0;
        for (y = 1; y < (m+1); y++ ) {

            for (x = 1; x < (n+1); x++) {

                updateVal = (   (SRC(x-1,y) + SRC(x+1,y))*cx +
                                (SRC(x,y-1) + SRC(x,y+1))*cy +
                                SRC(x,y)*cc - F(y-1, x-1)
                            )/cc;
                DST(x,y) = SRC(x,y) - relax*updateVal;
                error_in += updateVal*updateVal;
            }
        }
        error = sqrt(error_in)/(n*m);

        iterationCount++;

        // Swap the buffers
        tmp = u_old;
        u_old = u;
        u = tmp;
    }

    // Print MPI time
    printf( "Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount, (MPI_Wtime() - start_time) ); 
    MPI_Finalize();
    
    // Print system time
    int msec = (clock() - start) * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
    
    // Print the residual
    printf("Residual %12g\n",error);

    // Print the result
    //printBuf(u_old, n+2, m+2);
    
    // checkSolution() - calculate the error of the found solution
    error = 0;
    for (y = 1; y < (m+1); y++) {
        for (x = 1; x < (n+1); x++) {
            #define U(XX,YY) u_old[(YY)*(n+2)+(XX)]

            error_in = U(x,y) - (1.0-fX[x-1]*fX[x-1])*(1.0-fY[y-1]*fY[y-1]);
            error += error_in*error_in;
        }
    }

    printf("The error of the iterative solution is %g\n", sqrt(error)/(n*m));
    free(fX); free(fY);
    free(u); free(u_old);

    return 0;
}


// ---------------------------------------------------------------------------------------------------


void printHeader(int mode, int n, int m, double alpha, double relax, double tol, int mits){
    printf(" ________________________________________________________________________________ \n");
    printf("|                                                                                |\n");

    if(mode == 0){ // Serial Mode
        printf("|                             \033[33;1mSerial\033[0m \033[32;1mJacobi Method\033[0m                               |\n");
    }
    else if( mode == 1){ // MPI Mode
        printf("|                   \033[32;1mJacobi Method parallelization using\033[0m \033[33;1mMPI\033[0m                      |\n");
    }
    else if( mode == 2){ // MPI+OpenMP Hybrid Mode
        printf("|            \033[32;1mJacobi Method parallelization using\033[0m \033[33;1mMPI+OpenMP Hybrid\033[0m               |\n");

    }
    else{ // CUDA Mode
        printf("|                  \033[32;1mJacobi Method parallelization using\033[0m \033[33;1mCUDA\033[0m                      |\n");

    }

    printf("|________________________________________________________________________________|\n");
    printf("|                                                                                |\n");
    printf("|                 Parallel Systems 2020-2021, Main Assignment                    |\n");
    printf("|                    Ioannis Rovithakis - Georgios Galanis                       |\n");
    printf("|________________________________________________________________________________|\n");
    printf("|                                                                                |\n");
    printf("| Grid Dimentions: \033[33;1m%6ux%-6u\033[0m                                                 |\n", n,m);
    printf("| Input Alpha:     \033[33;1m%-6.2f\033[0m (Helmholtz Constant)                                   |\n", alpha);
    printf("| Input Relax:     \033[33;1m%-6.2f\033[0m (SOR parameter)                                        |\n", relax);
    printf("| Error Tolerance: \033[33;1m%-6.2g\033[0m                                                        |\n", tol);
    printf("| Max Iterations:  \033[33;1m%-6u\033[0m                                                        |\n", mits);
    printf("|________________________________________________________________________________|\n");
    printf("\n");
}

void printBuf(double *buf, int height, int width){
    int i=0, j=0;

    for(int index=0; index<(height*width); index++){
        printf(" %5.5f ", buf[index]);
        j++;
        if( j==height ){
            j=0;
            i++;
            printf("\n");
        }
    }printf("\n");
}

