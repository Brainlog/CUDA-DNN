#include <bits/stdc++.h>
using namespace std;
#include <chrono>

// each row in seperate block
__global__ void conv_kernel_v1(float* inp, float* out, int insize, float* kernel, int ksize){
    int row = blockIdx.x;
    int col = threadIdx.x;
    int outsize = insize - ksize + 1;
    float sum = 0;
    int lastpossindex = insize - ksize + 1;
    if(row >= lastpossindex || col >= lastpossindex){
        return;
    }
    for(int i=0; i<ksize; i++){
        for(int j=0; j<ksize; j++){
            sum += inp[(row+i)*insize + col+j] * kernel[i*ksize + j];
        }
    }
    out[row*outsize + col] = sum;
}

__global__ void conv_kernel_v2(float* inp, float* out, int insize, float* kernel, int ksize){
    int row = threadIdx.x;
    int col = threadIdx.y;
    int outsize = insize - ksize + 1;
    float sum = 0;
    int lastpossindex = insize - ksize + 1;
    if(row >= lastpossindex || col >= lastpossindex){
        return;
    }
    for(int i=0; i<ksize; i++){
        for(int j=0; j<ksize; j++){
            sum += inp[(row+i)*insize + col+j] * kernel[i*ksize + j];
        }
    }
    out[row*outsize + col] = sum;
}



int main(){
    int insize = 32;
    int ksize = 3;
    float* inp = new float[insize*insize];
    float* kernel = new float[ksize*ksize];
    float* out = new float[(insize-ksize+1)*(insize-ksize+1)];
    float* inp_d, *out_d, *kernel_d;
    cudaMalloc(&inp_d, insize*insize*sizeof(float));
    cudaMalloc(&out_d, (insize-ksize+1)*(insize-ksize+1)*sizeof(float));
    cudaMalloc(&kernel_d, ksize*ksize*sizeof(float));
    for(int i=0; i<insize; i++){
        for(int j=0; j<insize; j++){
            inp[i*insize + j] = i*insize + j;
        }
    }
    for(int i=0; i<ksize; i++){
        for(int j=0; j<ksize; j++){
            kernel[i*ksize + j] = i*ksize + j;
        }
    }
    cudaMemcpy(inp_d, inp, insize*insize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_d, kernel, ksize*ksize*sizeof(float), cudaMemcpyHostToDevice);
    // Measure Time : 
    // Method 1
    auto start1 = std::chrono::high_resolution_clock::now();
    conv_kernel_v1<<<insize-ksize+1, insize-ksize+1>>>(inp_d, out_d, insize, kernel_d, ksize);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = end1 - start1;    
    double duration_ms1 = duration1.count() * 1000.0;
    std::cout << duration_ms1 << endl;

    // Method 2
    auto start = std::chrono::high_resolution_clock::now();
    dim3 blocksize(insize-ksize+1,insize-ksize+1);
    conv_kernel_v2<<<1, blocksize>>>(inp_d, out_d, insize, kernel_d, ksize);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double duration_ms = duration.count() * 1000.0;
    std::cout << duration_ms << endl;
    

    return 0;
}
