# Parallel Programming Inference on CNN LeNet Architecture

## Installation Instructions on Laptop for running GPU code, Prerequisite: Laptop with NVIDIA GPU Driver:

Setting Driver    
```
Software Updates (Ubuntu) -> Additional Drivers -> Switch to 535
```

Install CUDA 11.8 on your Laptop in Conda Environment
```
make setcuda
```

## Execution Instructions for Code:
<!-- 
### Subtask-1:

```

``` -->
### Subtask-2:
Change the configurable parameters in the makefile under `sub2` and run `make sub2`. 
1. The command will be in the form of:
 ```
 ./subtask2 [function-choice: 1=convolution, 2=non-linear-activations, 3=subsampling, 4=converting a vector]
 ```
2. For convolution, the command would be changed to the following, where N = size of input matrix, M = size of kernel, P = padding length:
```
./subtask2 1 [N] [M] [P] [entries of matrix in row major order space separated] [entries of kernel in row major order space separated]
```


3. For activation, the command would be changed to the following, where N = size of input matrix: 
```
./subtask2 2 [0:relu, 1:tanh] [N] [entries of matrix in row major order space separated]
``` 

4. For pooling, the command would be changed to the following, where N = size of input matrix, M = size of kernel:

```
./subtask2 3 [0: maxpool, 1: avgpool] [N] [M] [entries of matrix in row major order space separated]
```

5. For softmax/sigmoid, the command would be the following, where N is the size of the input vector:

```
./subtask2 4 [0: sigmoid, 1: softmax] [N] [entries of input vector space separated]
```

The outputs will be printed in the file `output_subtask_2.txt`

### Subtask-3:

To run the Lenet architecture on the files, run the following lines:

```
1. python preprocess.py
2. ./subtask3 
```
The first line will read the images stored in `./mnist_test/test` and store them in `test_dataset.txt`. 

The second line will run the architecture on the images and print the top 5 output probabilities corresponding to each file in a file `output_subtask_3.txt`

### Subtask-4:

To run `subtask4.cu`, the command will be in the following format:

```
./subtask4 [1: with streams, 0: without streams]
```

The execution will print the top 5 output probabilities corresponding to each file in a file `output_subtask_3.txt`







