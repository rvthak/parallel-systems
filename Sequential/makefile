OUT=jacobi_serial

all: $(OUT)

$(OUT):
	mpicc $(OUT).c -o $(OUT).x -lm -O3

run: all
	qsub ./PBSseqJ.sh
