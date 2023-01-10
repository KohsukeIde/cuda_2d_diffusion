# cuda_c_2d_diffusion (ipynb)
## 2D Diffusion Simulation

This is a CUDA program that performs 2D diffusion on a grid of size `NX` x `NY` using the finite difference method. The program reads the initial grid values from a file and then performs a series of iterations to simulate the diffusion process.

### Compiling the code

To compile the code, you will need to have a CUDA-compatible GPU and a CUDA compiler installed on your system. You can then use the `nvcc` compiler to compile the code by running the following command:

```bash
nvcc main.cu diffusion2d.cu etc.cu bmp.cu -o main
```

This will compile the `main.cu` and `func.cu` files and create an executable file named `main`.

### Running the code

To run the code, you can use the following command:

```bash
./main
```

This will execute the `main` function, which initializes the grid and performs a series of iterations of the diffusion process. The program will output the current time and generate an image file at regular intervals. The program will continue until a certain number of iterations have been performed or until a maximum time has been reached.

### Configuration

You can modify the behavior of the program by modifying the following constants:

- `NX` and `NY`: These constants define the size of the grid in the `x` and `y` dimensions, respectively.
- `Lx` and `Ly`: These constants define the size of the grid in physical units in the `x` and `y` dimensions, respectively.
- `kappa`: This constant represents the diffusion coefficient.
- `nout`: This constant determines how often the program outputs the current time and generates an image file.

You can also modify the initial grid values by modifying the `initial` function. This function is called at the beginning of the program and sets the initial values of the grid based on a given function.

### Output

The program generates image files representing the state of the grid at regular intervals. The files are in BMP format and are named `f000.bmp`, `f001.bmp`, and so on. You can view these files using an image viewer or convert them to a different format using an image conversion tool.

The program also outputs the current time at regular intervals. The time is output to `stderr` and is expressed in units of the grid spacing squared divided by the diffusion coefficient. 

### Additional files

The program includes the following additional files:

- `main.cu`: This file is a mian file that simulates the diffusion of a 2D grid using the finite difference method.
- `diffusion.cu`: This file contains the implementation of the `diffusion_2d` function.
- `etc.cu`:This file contains several utility functions that are used by the main program.
- `bmp.cu`: This file contains the implementation of the `generate_bmp_image` function.



"The code in this project was developed based on the concepts and techniques described in the book 'はじめてのCUDAプログラミング' by 青木 尊之
 (2009)."
