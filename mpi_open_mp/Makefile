mpi_ex: mpi_convolution.o
	mpicc -o mpi_ex mpi_convolution.o -lm -fopenmp

mpi_convolution.o: mpi_convolution.c
	mpicc -c mpi_convolution.c -lm -fopenmp

clean:
	\rm -f *.o mpi_ex
