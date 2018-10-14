mpi_openmp_ex: mpi_open_mp_convolution.o
	mpicc -o mpi_openmp_ex mpi_open_mp_convolution.o -lm -fopenmp

mpi_open_mp_convolution.o: mpi_open_mp_convolution.c
	mpicc -c mpi_open_mp_convolution.c -lm -fopenmp

clean:
	\rm -f *.o mpi_openmp_ex
