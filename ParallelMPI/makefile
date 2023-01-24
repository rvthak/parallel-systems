OUT=jacobi_mpi

all: $(OUT)

$(OUT):
	mpicc $(OUT).c -o $(OUT).x -O3 -L/opt/mpiP-3.5/lib -lmpiP -lbfd -lunwind -lm

run: all
	qsub ./PBSmpiJ.sh
