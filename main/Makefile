

all:
	g++ -I ${mkEigenInc} -O3 main1.cpp -o main1
	mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis etest1.c -o etest1
	./main1 einstein.jpg