
// Ioannis Rovithakis | sdi1800164
// Georgios Galanis   | sdi1800024

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// UI/Print functions
void printHeader(int mode, int n, int m, double alpha, double relax, double tol, int mits, int comm_sz);
void printSqrErr();
void printBuf(double *buf, int height, int width);

// Simple input struct used to pass the input arguments to all the processes
typedef struct {
    int dim_x;
    int dim_y;
    double alpha;
    double relax;
    double tol;
    int max_its;
} Input;

// mpicc jacobi_mpi.c -o jacobi_mpi.x -lm -Wall -O3
int main(int argc, char **argv){

    int n, m, mits, iterationCount = 0, x, y, my_rank, comm_sz, msec, gsec, *f_index, x_off, y_off;
    double alpha, tol, relax, error = HUGE_VAL, *tmp, error_in = 0.0, updateVal, err_sum=0;
    Input input; MPI_Datatype tmpt;
    
    // Start MPI
    MPI_Init(&argc, &argv);
    MPI_Pcontrol(0);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz) ; 
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if( comm_sz!=4 && comm_sz!=9 && comm_sz!=16 && comm_sz!=25 && comm_sz!=36 && comm_sz!=49 && comm_sz!=64 && comm_sz!=80 ){ printf(" > Given process count outside the range of this assignment\n"); return 1;  }
    //___________________________________________________________________________
    
    // Create Input Struct-Datatype to pass the input arguments to all the processes
    int  block_lengths[6] = {1,1,1,1,1,1};
    MPI_Datatype types[6] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
    MPI_Aint   offsets[6] = { offsetof(Input, dim_x), offsetof(Input, dim_y), offsetof(Input, alpha  ),
                              offsetof(Input, relax), offsetof(Input, tol  ), offsetof(Input, max_its) };
    MPI_Datatype mpi_Input;
    MPI_Type_create_struct(6, block_lengths, offsets, types, &mpi_Input);
    MPI_Type_commit(&mpi_Input);

    // Read the Input Arguments and Broadcast them to all the processes
    if( my_rank == 0 ){
        scanf("%d,%d", &n, &m); scanf("%lf", &alpha);  scanf("%lf", &relax); scanf("%lf", &tol); scanf("%d", &mits);
        input.dim_x=n; input.dim_y=m; input.alpha=alpha; input.relax=relax; input.tol=tol; input.max_its=mits;
        printHeader(1, n, m, alpha, relax, tol, mits, comm_sz);

        if( n!=m ){ printSqrErr(); MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    MPI_Bcast(&input, 1, mpi_Input, 0, MPI_COMM_WORLD);
    //printHeader(1, input.dim_x, input.dim_y, input.alpha, input.relax, input.tol, input.max_its);
    n=input.dim_x, m=input.dim_y, alpha=input.alpha, relax=input.relax, tol=input.tol, mits=input.max_its;
    MPI_Type_free(&mpi_Input); 
    //___________________________________________________________________________

    // Create the needed Virtual Topology - 2D Cartesian Matrix (with reordering enabled)
    MPI_Comm MY_CART_COM;
    int dims[2] = {0, 0}, periods[2] = {0, 0};
    MPI_Dims_create(comm_sz, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &MY_CART_COM); 
    //if( my_rank==0 ){ printf(" Grid Dimentions: %dx%d\n\n", dims[0], dims[1]);}

    // Get the assigned dimentions
    int my_coords[2];
    MPI_Cart_coords(MY_CART_COM, my_rank, 2, my_coords);
    //printf(" (%d) : [%d, %d]\n", my_rank, my_coords[0], my_coords[1]); 

    // Get the process's neighbours
    int north_rank, south_rank, west_rank, east_rank;
    MPI_Cart_shift(MY_CART_COM, 0, 1, &north_rank, &south_rank);
    MPI_Cart_shift(MY_CART_COM, 1, 1, &west_rank, &east_rank);
    //printf(" (%d) North: %2d, South: %2d, West: %2d, East: %2d\n", my_rank, north_rank, south_rank, west_rank, east_rank);
    //___________________________________________________________________________

    // Allocate the main matrix
    double *u_global = NULL;
    if( my_rank == 0 ){
        // Only the "main" process allocates the main matrix (Inits everything to zero, including boundaries)
        if ( (u_global = (double*)calloc(n*m, sizeof(double))) == NULL ){ 
            printf(" (%d): Not enough memory for a %ix%i matrix\n", my_rank, n, m);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    //___________________________________________________________________________

    // Get the dimentions of the matrix sub-blocks based on the amount of processes
    int block_width  = n/dims[1];
    int block_height = m/dims[0];
    //if( my_rank==0 ){ printf(" Block Dimentions: %dx%d\n", block_width, block_height); printf("\n");}

    // Create a vector-based struct to represent the "main" matrix blocks
    MPI_Datatype mblock;
    MPI_Type_vector(block_height, block_width, block_width*dims[1], MPI_DOUBLE, &tmpt);
    MPI_Type_create_resized(tmpt, 0, sizeof(double), &mblock);
    MPI_Type_commit(&mblock);
    MPI_Type_free(&tmpt);

    // Create another vector-based struct to represent the "internal" matrix blocks (With padding to store halos)
    MPI_Datatype iblock;
    MPI_Type_vector(block_height, block_width, block_width+2, MPI_DOUBLE, &tmpt);
    MPI_Type_create_resized(tmpt, 0, sizeof(double), &iblock);
    MPI_Type_commit(&iblock);
    MPI_Type_free(&tmpt);
    // We needed two types of structs to correctly offset the sub-matrices, since our
    // main matrix is stored as a 1D matrix and not as a 2D matrix

    // Allocate the local block matrices used for calculations
    double *u, *u_old;
    if( (u = (double*)calloc((block_width+2)*(block_height+2),sizeof(double)) ) == NULL ||
    (u_old = (double*)calloc((block_width+2)*(block_height+2),sizeof(double)) ) == NULL){
        printf(" (%d): Not enough memory for a %ix%i block\n", my_rank, (dims[0]+2), (dims[1]+2));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate a small array to store the 'f' precaclulation array offsets used for each process
    if( (f_index = (int *)calloc(2, sizeof(int)))==NULL ){
        printf(" (%d): Not enough memory to store the f indexes\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Primary process splits the matrix into blocks and assigns them to processes
    int counts_send[comm_sz], displacements[comm_sz], f_offsets[comm_sz][2];
    if( my_rank==0 ){
        // For each block in the cartesian grid, find its displacement offset and its "f" offsets
        for(int j=0; j<dims[1]; j++){
            for(int i=0; i<dims[0]; i++){

                // Used to correctly split the main matrix between the different processes
                counts_send[i*dims[1]+j] = 1;
                displacements[i*dims[1]+j] = (dims[1]*block_width*block_height) * i + block_width * j;
                //printf(" [%d,%d] (%d) : displacement:%d\n", i,j, i*dims[1]+j, displacements[i*dims[1]+j]);

                // Used to send the correct offset to each process in order to use the correct f values
                f_offsets[i*dims[1]+j][0] = 1 + i * block_height ;
                f_offsets[i*dims[1]+j][1] = 1 + j * block_width ;
                //printf("              : f_offsets: y = %d, x = %d \n", f_offsets[i*dims[1]+j][0], f_offsets[i*dims[1]+j][1]);
            }
        }
    }

    // Scatter the blocks to their corresponding process
    MPI_Scatterv(u_global, counts_send, displacements, mblock, u+(block_width+2)+1,     1, iblock, 0, MPI_COMM_WORLD);
    MPI_Scatterv(u_global, counts_send, displacements, mblock, u_old+(block_width+2)+1, 1, iblock, 0, MPI_COMM_WORLD);
    MPI_Scatter(f_offsets, 2, MPI_INT, f_index, 2, MPI_INT, 0, MPI_COMM_WORLD);
    //printf(" (%d): y = %d, x = %d \n", my_rank, f_index[0], f_index[1]);
    //___________________________________________________________________________

    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(m-1);

    // Coefficients
    double cx = 1.0/(deltaX*deltaX);
    double cy = 1.0/(deltaY*deltaY);
    double cc = -2.0*cx-2.0*cy-alpha;
    //___________________________________________________________________________

    // Create "row" and "column" types that will be used for "halo" transfers
    MPI_Datatype block_row;
    MPI_Type_contiguous(block_width, MPI_DOUBLE, &tmpt);
    MPI_Type_create_resized(tmpt, 0, sizeof(double), &block_row);
    MPI_Type_commit(&block_row);
    MPI_Type_free(&tmpt);

    MPI_Datatype block_column;
    MPI_Type_vector(block_height, 1, block_width+2, MPI_DOUBLE, &tmpt);
    MPI_Type_create_resized(tmpt,0,sizeof(double),&block_column);
    MPI_Type_commit(&block_column);
    MPI_Type_free(&tmpt);
    //___________________________________________________________________________

    // Start Persistent communication since the neighbors never change
    // Create two groups of Presistent connections - One to send/receive using u and one for u_old since we switch them around 
    MPI_Request *send_reqs = (MPI_Request *)malloc(8*sizeof(MPI_Request));
    MPI_Request *recv_reqs = (MPI_Request *)malloc(8*sizeof(MPI_Request));
    if( send_reqs == NULL || recv_reqs == NULL ){ printf("Not enough memory to store the persistent connection requests\n");  MPI_Abort(MPI_COMM_WORLD, 1); }

    // Receive Group
    MPI_Recv_init( u_old+1                                              , 1, block_row    , north_rank , MPI_ANY_TAG, MPI_COMM_WORLD, &(recv_reqs[0]) );
    MPI_Recv_init( u_old+(block_width+2)*(block_height+2)-block_width-1 , 1, block_row    , south_rank , MPI_ANY_TAG, MPI_COMM_WORLD, &(recv_reqs[1]) );
    MPI_Recv_init( u_old+(block_width+2)                                , 1, block_column , west_rank  , MPI_ANY_TAG, MPI_COMM_WORLD, &(recv_reqs[2]) );
    MPI_Recv_init( u_old+(2*block_width+3)                              , 1, block_column , east_rank  , MPI_ANY_TAG, MPI_COMM_WORLD, &(recv_reqs[3]) );

    MPI_Recv_init( u+1                                              , 1, block_row    , north_rank , MPI_ANY_TAG, MPI_COMM_WORLD, &(recv_reqs[4]) );
    MPI_Recv_init( u+(block_width+2)*(block_height+2)-block_width-1 , 1, block_row    , south_rank , MPI_ANY_TAG, MPI_COMM_WORLD, &(recv_reqs[5]) );
    MPI_Recv_init( u+(block_width+2)                                , 1, block_column , west_rank  , MPI_ANY_TAG, MPI_COMM_WORLD, &(recv_reqs[6]) );
    MPI_Recv_init( u+(2*block_width+3)                              , 1, block_column , east_rank  , MPI_ANY_TAG, MPI_COMM_WORLD, &(recv_reqs[7]) );

    // Send Group
    MPI_Send_init( u_old+(block_width+3)                                  , 1, block_row    , north_rank , 0 , MPI_COMM_WORLD, &(send_reqs[0]) );
    MPI_Send_init( u_old+(block_width+2)*(block_height+2)-2*block_width-3 , 1, block_row    , south_rank , 1 , MPI_COMM_WORLD, &(send_reqs[1]) );
    MPI_Send_init( u_old+(block_width+3)                                  , 1, block_column , west_rank  , 2 , MPI_COMM_WORLD, &(send_reqs[2]) );
    MPI_Send_init( u_old+(2*block_width+2)                                , 1, block_column , east_rank  , 3 , MPI_COMM_WORLD, &(send_reqs[3]) );

    MPI_Send_init( u+(block_width+3)                                  , 1, block_row    , north_rank , 0 , MPI_COMM_WORLD, &(send_reqs[4]) );
    MPI_Send_init( u+(block_width+2)*(block_height+2)-2*block_width-3 , 1, block_row    , south_rank , 1 , MPI_COMM_WORLD, &(send_reqs[5]) );
    MPI_Send_init( u+(block_width+3)                                  , 1, block_column , west_rank  , 2 , MPI_COMM_WORLD, &(send_reqs[6]) );
    MPI_Send_init( u+(2*block_width+2)                                , 1, block_column , east_rank  , 3 , MPI_COMM_WORLD, &(send_reqs[7]) );
    //___________________________________________________________________________

    // Allocate the precomputation buffers (TO PARALLELIZE IF POSSIBLE)
    double *fX = (double *)malloc(n*sizeof(double));
    double *fY = (double *)malloc(m*sizeof(double));
    if (fX == NULL || fY == NULL ) {  printf("Not enough memory for fX, fY matrices\n");  MPI_Abort(MPI_COMM_WORLD, 1); }

    clock_t start = clock();

    // Pre-Compute fx, fy and f and store for future access - reuse
    for(y = 1; y < (m+1); y++ ){ fY[y-1] = yBottom + (y-1)*deltaY; }
    for(x = 1; x < (n+1); x++ ){ fX[x-1] = xLeft   + (x-1)*deltaX; }
    //___________________________________________________________________________

    // Sync the processes before timing them
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Pcontrol(1);

    // Start the timers
    double start_time, stop_time, global_time;
    start_time = MPI_Wtime();    

    // Main loop of the iterative process
    while (iterationCount < mits && error > tol ) {  
        #define SRC(XX,YY) u_old[(YY)*(block_width+2)+(XX)]
        #define DST(XX,YY) u[(YY)*(block_width+2)+(XX)]
        #define F(XX,YY) ( -alpha*(1.0-fX[XX]*fX[XX])*(1.0-fY[YY]*fY[YY]) - 2.0*(1.0-fX[XX]*fX[XX]) - 2.0*(1.0-fY[YY]*fY[YY]) )

        // Receive the needed halo points
        MPI_Startall(4, recv_reqs + 4*(iterationCount%2) );
        // Send your most recent halo points
        MPI_Startall(4, send_reqs + 4*(iterationCount%2) );
        
        // Calculate the values of the inner, non halo-dependent points
        // We can calculate them without any problem while we wait for the halo points to arrive
        error_in = 0; 

        y_off = f_index[0]+1;
        for (y = 2; y < block_height ; y++ ) {
            x_off = f_index[1]+1;
            for (x = 2; x < block_width ; x++) {
                updateVal = (   (SRC(x-1,y) + SRC(x+1,y))*cx +
                                (SRC(x,y-1) + SRC(x,y+1))*cy +
                                SRC(x,y)*cc - F(y_off-1, x_off-1)
                            )/cc;
                DST(x,y) = SRC(x,y) - relax*updateVal;
                error_in += updateVal*updateVal;
                x_off++;
            }
            y_off++;
        }

        // Wait for all the halo point to arrive
        MPI_Waitall(4, recv_reqs + 4*(iterationCount%2) , MPI_STATUSES_IGNORE);

        // Calculate the values of the outer, halo-dependent points
        // (One "for" for each halo-dependent row/column)
        y_off = f_index[0];
        for(y=1; y < block_height+1; y++){ // x==1
            updateVal = (   (SRC(0,y) + SRC(2,y))*cx +
                            (SRC(1,y-1) + SRC(1,y+1))*cy +
                            SRC(1,y)*cc - F(y_off-1, f_index[1]-1)
                        )/cc;
            DST(1,y) = SRC(1,y) - relax*updateVal;
            error_in += updateVal*updateVal;
            y_off++;
        }

        y_off = f_index[0];
        for(y=1; y < block_height+1; y++){ // x==block_width
            updateVal = (   (SRC(block_width-1,y) + SRC(block_width+1,y))*cx +
                            (SRC(block_width,y-1) + SRC(block_width,y+1))*cy +
                            SRC(block_width,y)*cc - F(y_off-1, f_index[1]+block_width-2)
                        )/cc;
            DST(block_width,y) = SRC(block_width,y) - relax*updateVal;
            error_in += updateVal*updateVal;
            y_off++;
        }

        x_off = f_index[1]+1;
        for(x=2; x < block_width; x++){ // y==1
            updateVal = (   (SRC(x-1,1) + SRC(x+1,1))*cx +
                            (SRC(x,0) + SRC(x,2))*cy +
                            SRC(x,1)*cc - F(f_index[0]-1, x_off-1) 
                        )/cc;
            DST(x,1) = SRC(x,1) - relax*updateVal;
            error_in += updateVal*updateVal;
            x_off++;
        }

        x_off = f_index[1]+1;
        for(x=2; x < block_width; x++){ // y==block_height
            updateVal = (   (SRC(x-1,block_height) + SRC(x+1,block_height))*cx +
                            (SRC(x,block_height-1) + SRC(x,block_height+1))*cy +
                            SRC(x,block_height)*cc - F(f_index[0]+block_height-2, x_off-1)
                        )/cc;
            DST(x,block_height) = SRC(x,block_height) - relax*updateVal;
            error_in += updateVal*updateVal;
            x_off++;
        }

        iterationCount++;

        // Swap the buffers
        tmp = u_old;
        u_old = u;
        u = tmp;

        // Finally calculate the error using reduce
        MPI_Allreduce(&error_in, &err_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        error = sqrt(err_sum)/(n*m);
        
        // Wait until all your halos have been sent for this iteration
        MPI_Waitall(4, send_reqs + 4*(iterationCount%2) , MPI_STATUSES_IGNORE);
    }
    MPI_Pcontrol(0);
    
    // Stop the timers
    stop_time = MPI_Wtime() - start_time;
    msec = (clock() - start) * 1000 / CLOCKS_PER_SEC;

    // Get the max time between all the processes and print it
    MPI_Reduce(&stop_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&msec, &gsec, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if( my_rank==0 ){
        printf(" ________________________________________________________________________________ \n");
        printf("|                                                                                |\n");
        printf("| \033[36;1mIterations\033[0m: \033[33;1m%-4d \033[0m                                                              |\n", iterationCount);
        printf("| \033[36;1mElapsed MPI Wall Time\033[0m: \033[33;1m%-8.4f \033[0m                                               |\n", global_time);
        printf("| \033[36;1mTime taken\033[0m: \033[33;1m%-4d seconds and %-3d milliseconds \033[0m                                 |\n", gsec/1000, gsec%1000);
        printf("| \033[36;1mResidual\033[0m: \033[33;1m%-15g \033[0m                                                     |  \n",error);
    }
    //___________________________________________________________________________

    // Gather the resulting block matrices and check the solution
    MPI_Gatherv(u_old+(block_width+2)+1, 1, iblock, u_global, counts_send, displacements, mblock, 0, MPI_COMM_WORLD);
    free(u);free(u_old);

    // checkSolution() - Calculate the error of the found solution
    if( my_rank == 0 ){
        error = 0;
        for (y = 0; y < m; y++) {
            #define U(XX,YY) u_global[(YY)*(n)+(XX)]
            for (x = 0; x < n; x++) {
                error_in = U(x,y) - (1.0-fX[x]*fX[x])*(1.0-fY[y]*fY[y]);
                error += error_in*error_in;
            }
        }
        printf("| The \033[36;1mError\033[0m of the iterative solution is \033[33;1m%-15g\033[0m                         |\n", sqrt(error)/(n*m));
        printf("|________________________________________________________________________________|\n\n");
    }

    // Print the result
    //if( my_rank == 0 ){ printBuf(u_global, n,m); }

    // Garbage Collection
    free(fX); free(fY); free(f_index);
    MPI_Type_free(&mblock); MPI_Type_free(&iblock);
    MPI_Type_free(&block_row); MPI_Type_free(&block_column);
    MPI_Comm_free(&MY_CART_COM);

    for( int i=0; i<8; i++){
        MPI_Request_free(&(send_reqs[i]));
        MPI_Request_free(&(recv_reqs[i]));
    } free(send_reqs); free(recv_reqs);

    if( my_rank==0 ){ free(u_global); }
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}


// ---------------------------------------------------------------------------------------------------


void printHeader(int mode, int n, int m, double alpha, double relax, double tol, int mits, int comm_sz){
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
    printf("| Processes:       \033[33;1m%-6u\033[0m                                                        |\n", comm_sz);
    printf("|________________________________________________________________________________|\n");
    printf("\n");
}

void printBuf(double *buf, int height, int width){
    int i=0, j=0;

    for(int index=0; index<(height*width); index++){
        printf(" %5.4f ", buf[index]);
        j++;
        if( j==height ){
            j=0;
            i++;
            printf("\n");
        }
    }printf("\n");
}

void printSqrErr(){
    printf(" ________________________________________________________________________________ \n");
    printf("|                                                                                |\n");
    printf("|                                  \033[31;1mFatal Error:\033[0m                                  |\n");
    printf("| > The given matrix is not Sqare (n==m). This program version only supports     |\n");
    printf("|   Square matrices. Please input a Square matrix.                               |\n");
    printf("|________________________________________________________________________________|\n");
}