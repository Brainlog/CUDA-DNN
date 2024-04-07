#include <bits/stdc++.h>
using namespace std;

__global__ void conv_kernel_p1(float *inp, float *out, int insize, float *kernel, int ksize, int inchannels, int kchannels, int flag)
{
    int inchannel = blockIdx.y;
    int kchannel = blockIdx.x;
    int outchannel = (kchannel)*inchannels + inchannel;
    int row = threadIdx.x;
    int col = threadIdx.y;
    int outsize = insize - ksize + 1;
    float sum = 0;
    if (inchannel < inchannels && kchannel < kchannels && row < outsize && col < outsize)
    {
        for (int i = 0; i < ksize; i++)
        {
            for (int j = 0; j < ksize; j++)
            {
                sum += inp[inchannel * insize * insize + (row + i) * insize + col + j] * kernel[outchannel * ksize * ksize + i * ksize + j];
            }
        }
        if (flag == 0)
            out[outchannel  + (row * outsize + col)*(kchannels*inchannels)] = sum;
        else
            out[outchannel * outsize * outsize + row * outsize + col] = sum;
    }
}

__global__ void conv_kernel_p2(float *inp, float *out, int kchannels, int inchannels, int insize, float *bias)
{
    int kchannel = blockIdx.x;
    int row = threadIdx.x;
    int col = threadIdx.y;
    float temp = 0;
    if (kchannel < kchannels && row < insize && col < insize)
    {
        for (int i = 0; i < inchannels; i++)
        {
            int currchannel = inchannels * kchannel + i;
            temp += inp[currchannel  + (row * insize + col)*(kchannels*inchannels)];
        }
        out[kchannel * insize * insize + row * insize + col] = temp + bias[kchannel];
    }
}

__global__ void maxpool_kernel(float *inp, float *out, int insize, int ksize, int stride, int inchannels)
{
    int outsize = (insize - ksize) / stride + 1;
    int inchannel = blockIdx.x;
    int row = threadIdx.x;
    int col = threadIdx.y;
    int outchannel = inchannel;
    float maxval = 0;
    if (inchannel < inchannels && row % stride == 0 && col % stride == 0 && row + ksize - 1 < insize && col + ksize - 1 < insize)
    {
        int newrow = row / stride;
        int newcol = col / stride;
        for (int i = 0; i < ksize; i++)
        {
            for (int j = 0; j < ksize; j++)
            {
                maxval = max(maxval, inp[inchannel * insize * insize + (row + i) * insize + col + j]);
            }
        }
        out[outchannel * outsize * outsize + newrow * outsize + newcol] = maxval;
    }
}

__global__ void avgpool_kernel(float *inp, float *out, int insize, int ksize, int stride, int inchannels)
{
    int outsize = (insize - ksize) / stride + 1;
    int inchannel = blockIdx.x;
    int row = threadIdx.x;
    int col = threadIdx.y;
    int outchannel = inchannel;
    float maxval = 0;
    if (inchannel < inchannels && row % stride == 0 && col % stride == 0 && row + ksize - 1 < insize && col + ksize - 1 < insize)
    {
        int newrow = row / stride;
        int newcol = col / stride;
        for (int i = 0; i < ksize; i++)
        {
            for (int j = 0; j < ksize; j++)
            {
                maxval = maxval + inp[inchannel * insize * insize + (row + i) * insize + col + j];
            }
        }
        out[outchannel * outsize * outsize + newrow * outsize + newcol] = maxval/(ksize*ksize);
    }
}

__global__ void fc_kernel(float *inp, float *out, float *weight, float *bias, int insize, int outsize)
{
    int row = threadIdx.x;
    float sum = 0;
    for (int i = 0; i < insize; i++)
    {
        sum += inp[i] * weight[row * insize + i];
    }
    out[row] = sum + bias[row];
}


__global__ void relu_kernel(float *inp, float *out, int insize)
{
    int row = threadIdx.x;
    int col = threadIdx.y;
    float init = 0.0;
    out[row * insize + col] = max(inp[row * insize + col], init);
        
}

__global__ void tanh_kernel(float *inp, float *out, int insize)
{
    int row = threadIdx.x;
    int col = threadIdx.y;
    float ex_1 = exp(inp[row * insize + col]);
    float ex_2 = 1.0 / ex_1;
    out[row * insize + col] = (ex_1 - ex_2) / (ex_1 + ex_2);
}


void softmax(float *vector, float *final_vector, int size)
{
    float denom = 0;
    float max = vector[0];
    for (int i = 1; i < size; i++)
    {
        if (vector[i] > max)
            max = vector[i];
    }
    for (int i = 0; i < size; i++)
        vector[i] -= max;
    for (int i = 0; i < size; i++)
        denom += exp(vector[i]);
    for (int i = 0; i < size; i++)
        final_vector[i] = exp(vector[i]) / denom;
}

void sigmoid(float *vector, float *final_vector, int size)
{
    for(int i=0; i < size; i++){
        final_vector[i] = 1.0 / (1.0 + exp(vector[i]));
    }
}



int main(int argc, char **argv){
    if(argc < 2){
        cout << "Please provide input parameters" << endl; 
        return 0;
    }
    int mode = atoi(argv[1]);
    if(mode == 1){    // Convolution

        // Get parameters for convolution

        int insize = atoi(argv[2]);
        int ksize = atoi(argv[3]);
        int padding = atoi(argv[4]);
        int inchannels = 1;
        int kchannels = 1;
        int flag = 1;

        // Read matrix and kernel inputs

        float *inp = (float *)malloc(insize * insize * sizeof(float));
        float *kernel = (float *)malloc(ksize * ksize * sizeof(float));

        for(int i = 5; i < 5 + insize * insize; i++)
        {
            inp[i - 5] = atof(argv[i]);
        }

        for(int i = 5 + insize * insize; i < 5 + insize * insize + ksize * ksize; i++)
        {
            kernel[i - 5 - insize * insize] = atof(argv[i]);
        }

        // Make padded matrix

        int newsize = insize + 2*padding;
        float *padded_inp = (float *)malloc(newsize * newsize * sizeof(float));
        for(int i = 0; i < newsize; i++)
        {
            for(int j = 0; j < newsize; j++)
            {
                if(i < padding || i >= padding + insize || j < padding || j >= padding + insize)
                {
                    padded_inp[i * newsize + j] = 0;
                }
                else
                {
                    padded_inp[i * newsize + j] = inp[(i - padding) * insize + j - padding];
                }
            }
        }

        // Allocate memory on device 

        float *d_inp;
        float *d_kernel;
        cudaMalloc(&d_inp, newsize * newsize * sizeof(float));
        cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

        // Copy data to device

        cudaMemcpy(d_inp, padded_inp, newsize * newsize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel, ksize * ksize * sizeof(float), cudaMemcpyHostToDevice);

        // Allocate memory for output

        int outsize = newsize - ksize + 1;

        float *out = (float *)malloc(outsize * outsize * sizeof(float));
        float *d_out;
        cudaMalloc(&d_out, outsize * outsize * sizeof(float));

        // Convolution kernel:
         
        dim3 threads1(outsize, outsize);
        dim3 blocks1(1, 1);
        conv_kernel_p1<<<blocks1, threads1>>>(d_inp, d_out, newsize, d_kernel, ksize, inchannels, kchannels, flag);

        // Bring output to host

        cudaMemcpy(out, d_out, outsize * outsize * sizeof(float), cudaMemcpyDeviceToHost);

        // Print output

        ofstream out_file;
        out_file.open("output_subtask_2.txt");

        for(int i = 0; i < outsize; i++)
        {
            for(int j = 0; j < outsize; j++)
            {
                out_file << out[i * outsize + j] << " ";
            }
            out_file << endl;
        }

        out_file.close();

        cudaFree(d_inp);
        cudaFree(d_kernel);
        cudaFree(d_out);

        return 0;
    }
    if(mode == 2){    // Activation

        int mode_of_activation = atoi(argv[2]);
        int insize = atoi(argv[3]);
        float *inp = (float *)malloc(insize * insize * sizeof(float));
        for(int i = 4; i < 4 + insize * insize; i++)
        {
            inp[i - 4] = atof(argv[i]);
        }
        float *out = (float *)malloc(insize * insize * sizeof(float));
        float *d_inp;
        float *d_out;

        cudaMalloc(&d_inp, insize * insize * sizeof(float));
        cudaMalloc(&d_out, insize * insize * sizeof(float));
        cudaMemcpy(d_inp, inp, insize * insize * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blocks1(1,1);
        dim3 threads1(insize, insize);
        
        if(mode_of_activation == 0){
            relu_kernel<<<blocks1, threads1>>>(d_inp, d_out, insize);
        }
        else{
            tanh_kernel<<<blocks1, threads1>>>(d_inp, d_out, insize);
        }

        cudaMemcpy(out, d_out, insize * insize * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_inp);
        cudaFree(d_out);

        ofstream out_file;
        out_file.open("output_subtask_2.txt");

        for(int i=0; i<insize; i++){
            for(int j=0; j<insize; j++){
                out_file << out[i * insize + j] << " ";
            }
            out_file << endl;
        }
        out_file.close();

        return 0;
    }
    if(mode == 3){    // Subsampling
        int mode_of_subsampling = atoi(argv[2]);
        int insize = atoi(argv[3]);
        int ksize = atoi(argv[4]);
        float *inp = (float *)malloc(insize * insize * sizeof(float));
        for(int i = 5; i < 5 + insize * insize; i++)
        {
            inp[i - 5] = atof(argv[i]);
        }
        int outsize = (insize - ksize) / 2 + 1;
        float *out = (float *)malloc(outsize * outsize * sizeof(float));
        float *d_inp;
        float *d_out;
        cudaMalloc(&d_inp, insize * insize * sizeof(float));
        cudaMalloc(&d_out, outsize * outsize * sizeof(float));
        cudaMemcpy(d_inp, inp, insize * insize * sizeof(float), cudaMemcpyHostToDevice);
        if(mode_of_subsampling == 0){
            dim3 threads1(insize, insize);
            dim3 blocks1(1, 1);
            maxpool_kernel<<<blocks1, threads1>>>(d_inp, d_out, insize, ksize, 2, 1);
            cudaMemcpy(out, d_out, outsize * outsize * sizeof(float), cudaMemcpyDeviceToHost);
        }
        else{
            dim3 threads1(insize, insize);
            dim3 blocks1(1, 1);
            avgpool_kernel<<<blocks1, threads1>>>(d_inp, d_out, insize, ksize, 2, 1);
            cudaMemcpy(out, d_out, outsize * outsize * sizeof(float), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_inp);
        cudaFree(d_out);

        ofstream out_file;
        out_file.open("output_subtask_2.txt");
        for(int i = 0; i < outsize; i++)
        {
            for(int j = 0; j < outsize; j++)
            {
                out_file << out[i * outsize + j] << " ";
            }
            out_file << endl;
        }
        out_file.close();

        return 0;
    }
    if(mode == 4){    // Softmax
        int soft_mode = atoi(argv[2]);
        
        int insize = atoi(argv[3]);
        float *inp = (float *)malloc(insize * sizeof(float));
        for(int i = 4; i < 4 + insize; i++)
        {
            inp[i - 4] = atof(argv[i]);
        }

        float *out = (float *)malloc(insize * sizeof(float));

        if(soft_mode == 1) softmax(inp, out, insize);
        else sigmoid(inp, out, insize);

        ofstream out_file;
        out_file.open("output_subtask_2.txt");
        for(int i=0; i<insize; i++){
            out_file << out[i] << " ";
        }
        out_file << endl;
        out_file.close();
        return 0;
    }
}

