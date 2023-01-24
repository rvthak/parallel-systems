OUT=jacobi_mpiH

all: $(OUT)

$(OUT):
	mpicc $(OUT).c -o $(OUT).x -O3 -L/opt/mpiP-3.5/lib -lmpiP -lbfd -lunwind -lm -fopenmp

run: all
	qsub ./PBShybridJ.sh
