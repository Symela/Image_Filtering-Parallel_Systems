mpi_ex: mpi_convolution.o
	mpicc -o mpi_ex mpi_convolution.o -lm

mpi_convolution.o: mpi_convolution.c
	mpicc -c mpi_convolution.c -lm

clean:
	\rm -f *.o mpi_ex
