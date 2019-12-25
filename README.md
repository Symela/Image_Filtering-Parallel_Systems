# Image_Filtering


## How to run with MPI



```bash
make

mpirexec -np <proc_num> ./mpi_ex <image> <image_type> <height> <width> <loops>
```

## How to run with MPI-OPENMP



```bash
make

mpirexec -np <proc_num> ./mpi_openmp_ex <image> <image_type> <height> <width> <loops>
```

## How to run with CUDA



```bash
make

./cuda_ex <image> <image_type> <height> <width> <loops> <blocksize>
```

## Results

#### GREY Image
![Grey Image](results/grey.png)

##### Grey Image 10 loops
![Grey Image](results/grey10.png)

##### Grey Image 50 loops
![Grey Image](results/grey50.png)

##### Grey Image 90 loops
![Grey Image](results/grey90.png)

---


#### RGB Image
![Grey Image](results/rgb.png)

##### Grey Image 10 loops
![Grey Image](results/rgb10.png)

##### Grey Image 50 loops
![Grey Image](results/rgb50.png)

##### Grey Image 90 loops
![Grey Image](results/rgb90.png)
