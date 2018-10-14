cuda_ex: filter_image.o cuda_functions.o
	nvcc -o cuda_ex filter_image.o cuda_functions.o -O2 -lm

filter_image.o: filter_image.c
	gcc -c filter_image.c -O2

cuda_functions.o: cuda_functions.cu
	nvcc -c cuda_functions.cu

clean:
	\rm -f *.o cuda_ex
