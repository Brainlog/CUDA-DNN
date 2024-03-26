#include <bits/stdc++.h>
using namespace std;
#include <chrono>

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

__global__ void conv_kernel_p1(float *inp, float *out, int insize, float *kernel, int ksize, int inchannels, int kchannels, float *bias, int flag)
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
            out[outchannel + (row * outsize + col) * (kchannels * inchannels)] = sum;
        else
            out[outchannel * outsize * outsize + row * outsize + col] = sum + bias[outchannel];
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
            temp += inp[currchannel + (row * insize + col) * (kchannels * inchannels)];
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

int main()
{
    // File extraction
    ofstream logger("./log.txt");
    ifstream conv1;
    conv1.open("./trained_weights/conv1.txt");
    ifstream conv2;
    conv2.open("./trained_weights/conv2.txt");
    ifstream conv3;
    conv3.open("./trained_weights/fc1.txt");
    ifstream fc2;
    fc2.open("./trained_weights/fc2.txt");

    // Conv1
    // Total filters = 20 Kernel size = 5 Input channels = 1 Output channels = 20 Input size = 28 Output size = 24 Bias = 20
    float *conv1_kernel = new float[20 * 5 * 5];
    float *conv1_bias = new float[20];

    // Reading weights and biases
    for (int i = 0; i < 20 * 5 * 5; i++)
    {
        conv1 >> conv1_kernel[i];
    }
    for (int i = 0; i < 20; i++)
    {
        conv1 >> conv1_bias[i];
    }

    // Conv2
    // assumed that filter[i][j] is the jth filter of the ith output channel
    // Total filters = 50x20, Kernel size = 5, Input channels = 20, Output channels = 50, Input size = 24, Output size = 8, Bias = 50
    float *conv2_kernel = new float[50 * 20 * 5 * 5];
    float *conv2_bias = new float[50];

    // Reading weights and biases
    for (int i = 0; i < 50 * 20 * 5 * 5; i++)
    {
        conv2 >> conv2_kernel[i];
    }
    for (int i = 0; i < 20; i++)
    {
        conv2 >> conv2_bias[i];
    }

    // Conv3
    // Total filters = 500, Kernel size = 4, Input channels = 50, Output channels = 500, Input size = 4, Output size = 1, Bias = 500
    float *conv3_kernel = new float[500 * 50 * 4 * 4];
    float *conv3_bias = new float[500];

    // Reading weights and biases
    for (int i = 0; i < 500 * 50 * 4 * 4; i++)
    {
        conv3 >> conv3_kernel[i];
    }
    for (int i = 0; i < 500; i++)
    {
        conv3 >> conv3_bias[i];
    }

    // FC2
    // Total weights = 10x500, Input size = 500, Output size = 10, Bias = 10
    float *fc2_weight = new float[10 * 500];
    float *fc2_bias = new float[10];

    // Reading weights and biases
    for (int i = 0; i < 10 * 500; i++)
    {
        fc2 >> fc2_weight[i];
    }
    for (int i = 0; i < 10; i++)
    {
        fc2 >> fc2_bias[i];
    }

    // input dataset and labels
    ifstream dataset;
    dataset.open("./test_dataset.txt");
    ifstream labels;
    labels.open("./test_labels.txt");
    float *inpu = new float[10000 * 28 * 28];
    int *label = new int[10000];
    for (int i = 0; i < 10000; i++)
    {
        labels >> label[i];
        for (int j = 0; j < 28 * 28; j++)
        {
            dataset >> inpu[i * 28 * 28 + j];
        }
    }
    // Device memory allocation for weights and biases
    float *d_conv1_kernel;
    float *d_conv1_bias;
    float *d_conv2_kernel;
    float *d_conv2_bias;
    float *d_conv3_kernel;
    float *d_conv3_bias;
    float *d_fc2_weight;
    float *d_fc2_bias;

    cudaMalloc(&d_conv1_kernel, 20 * 5 * 5 * sizeof(float));
    cudaMalloc(&d_conv1_bias, 20 * sizeof(float));
    cudaMalloc(&d_conv2_kernel, 50 * 20 * 5 * 5 * sizeof(float));
    cudaMalloc(&d_conv2_bias, 50 * sizeof(float));
    cudaMalloc(&d_conv3_kernel, 500 * 50 * 4 * 4 * sizeof(float));
    cudaMalloc(&d_conv3_bias, 500 * sizeof(float));
    cudaMalloc(&d_fc2_weight, 10 * 500 * sizeof(float));
    cudaMalloc(&d_fc2_bias, 10 * sizeof(float));
    cudaMemcpy(d_conv1_kernel, conv1_kernel, 20 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_bias, conv1_bias, 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_kernel, conv2_kernel, 50 * 20 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_bias, conv2_bias, 50 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv3_kernel, conv3_kernel, 500 * 50 * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv3_bias, conv3_bias, 500 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_weight, fc2_weight, 10 * 500 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_bias, fc2_bias, 10 * sizeof(float), cudaMemcpyHostToDevice);

    // Inference details
    int batch = 1000;
    int start = 0;
    int count = 0;
    int ksize1 = 5;
    int insize1 = 28;
    int inchannels1 = 1;
    int kchannels1 = 20;
    dim3 threads1(24, 24);
    dim3 blocks1(20, 1);
    int flag1 = 1;
    int ksize2 = 2;
    int stride2 = 2;
    int insize2 = 24;
    int inchannels2 = 20;
    dim3 threads2(24, 24);
    dim3 blocks2(20);
    int ksize3 = 5;
    int insize3 = 12;
    int inchannels3 = 20;
    int kchannels3 = 50;
    dim3 threads3(8, 8);
    dim3 threads3_2(8, 8);
    dim3 blocks3(50, 20);
    int ksize4 = 2;
    int stride4 = 2;
    int insize4 = 8;
    int inchannels4 = 50;
    dim3 threads4(8, 8);
    dim3 blocks4(50);
    int ksize5 = 4;
    int insize5 = 4;
    int inchannels5 = 50;
    int kchannels5 = 500;
    dim3 threads5(1, 1);
    dim3 threads5_2(1, 1);
    dim3 blocks5(500, 50);
    int insize6 = 500;
    int outsize6 = 10;
    dim3 threads6(10);

    // Memory allocation for input and output
    // float *inp = new float[28 * 28];
    float **inp = new float *[batch];
    for(int i = 0; i < batch; i++){
        inp[i] = new float[28 * 28];
    }
    float **d_inp = new float *[batch];
    float **d_out1_p1 = new float *[batch];
    float **d_out2 = new float *[batch];
    float **d_out3_p1 = new float *[batch];
    float **d_out3_p2 = new float *[batch];
    float **d_out4 = new float *[batch];
    float **d_out5_p1 = new float *[batch];
    float **d_out5_p2 = new float *[batch];
    float **d_out6 = new float *[batch];
    for (int i = 0; i < batch; i++)
    {
        cudaMalloc(&d_inp[i], 28 * 28 * sizeof(float));
        cudaMalloc(&d_out1_p1[i], 20 * 24 * 24 * sizeof(float));
        cudaMalloc(&d_out2[i], 20 * 12 * 12 * sizeof(float));
        cudaMalloc(&d_out3_p1[i], 20 * 50 * 8 * 8 * sizeof(float));
        cudaMalloc(&d_out3_p2[i], 50 * 8 * 8 * sizeof(float));
        cudaMalloc(&d_out4[i], 50 * 4 * 4 * sizeof(float));
        cudaMalloc(&d_out5_p1[i], 50 * 500 * sizeof(float));
        cudaMalloc(&d_out5_p2[i], 500 * sizeof(float));
        cudaMalloc(&d_out6[i], 10 * sizeof(float));
    }
    cudaDeviceSynchronize();
    for(int i = 0; i < batch; i++){
        for(int j = 0; j < 28 * 28; j++){
            inp[i][j] = inpu[i * 28 * 28 + j];
        }
    }
    for(int i = 0; i < batch; i++){
        cudaMemcpy(d_inp[i], inp[i], 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);
    }
    // count time
    auto str = std::chrono::high_resolution_clock::now();
    cudaStream_t* streams = new cudaStream_t[batch];
    for(int i = 0; i < batch; i++){
        cudaStreamCreate(&streams[i]);
    }

    // Inference Loop
    for (int i = start; i < start + batch; i++)
    {
        // Dimensions of all outputs :
        // Conv1 : 20x24x24, Pool1 : 20x12x12, Conv2 : 50x8x8, Pool2 : 50x4x4, Conv3 : 500, FC2 : 10
        // Perform Inference
        // Conv1
        conv_kernel_p1<<<blocks1, threads1, 0, streams[i]>>>(d_inp[i], d_out1_p1[i], insize1, d_conv1_kernel, ksize1, inchannels1, kchannels1, d_conv1_bias, flag1);

        // Pool1
        maxpool_kernel<<<blocks2, threads2, 0, streams[i]>>>(d_out1_p1[i], d_out2[i], insize2, ksize2, stride2, inchannels2);

        // Conv2
        conv_kernel_p1<<<blocks3, threads3, 0, streams[i]>>>(d_out2[i], d_out3_p1[i], insize3, d_conv2_kernel, ksize3, inchannels3, kchannels3, d_conv2_bias, 0);
        conv_kernel_p2<<<50, threads3_2, 0, streams[i]>>>(d_out3_p1[i], d_out3_p2[i], kchannels3, inchannels3, insize3 - ksize3 + 1, d_conv2_bias);

        // Pool2
        maxpool_kernel<<<blocks4, threads4, 0 ,streams[i]>>>(d_out3_p2[i], d_out4[i], insize4, ksize4, stride4, inchannels4);

        // Conv3
        conv_kernel_p1<<<blocks5, threads5, 0, streams[i]>>>(d_out4[i], d_out5_p1[i], insize5, d_conv3_kernel, ksize5, inchannels5, kchannels5, d_conv3_bias, 0);
        conv_kernel_p2<<<500, threads5_2, 0, streams[i]>>>(d_out5_p1[i], d_out5_p2[i], kchannels5, inchannels5, insize5 - ksize5 + 1, d_conv3_bias);

        // FC2
        fc_kernel<<<1, threads6, 0, streams[i]>>>(d_out5_p2[i], d_out6[i], d_fc2_weight, d_fc2_bias, insize6, outsize6);
    }
    cudaDeviceSynchronize();
    // // Probabilities
    float* maxi_indices = new float[batch];
    float** out6 = new float*[batch];
    for(int i = 0; i < batch; i++){
        out6[i] = new float[10];
        cudaMemcpy(out6[i], d_out6[i], 10 * sizeof(float), cudaMemcpyDeviceToHost);
        float *final_out = new float[10];
        softmax(out6[i], final_out, 10);
        int max_index = 0;
        for (int j = 0; j < 10; j++)
        {
            if (final_out[j] > final_out[max_index])
            {
                max_index = j;
            }
        }
        maxi_indices[i] = max_index;
        if(label[i] == max_index){
            count++;
        }
    } 


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - str);
    std::cout << "Total Time : " << duration.count() << "\n";
    std::cout << "Accuracy : " << count << " / " << batch << endl;

    cudaFree(d_inp);
    cudaFree(d_out1_p1);
    cudaFree(d_out2);
    cudaFree(d_out3_p1);
    cudaFree(d_out3_p2);
    cudaFree(d_out4);
    cudaFree(d_out5_p1);
    cudaFree(d_out5_p2);
    cudaFree(d_out6);

    // Free the memory of weights
    cudaFree(d_conv1_kernel);
    cudaFree(d_conv1_bias);
    cudaFree(d_conv2_kernel);
    cudaFree(d_conv2_bias);
    cudaFree(d_conv3_kernel);
    cudaFree(d_conv3_bias);
    cudaFree(d_fc2_weight);
    cudaFree(d_fc2_bias);

    // Free host memory
    delete[] inpu;
    delete[] label;
    delete[] conv1_kernel;
    delete[] conv1_bias;
    delete[] conv2_kernel;
    delete[] conv2_bias;
    delete[] conv3_kernel;
    delete[] conv3_bias;
    delete[] fc2_weight;
    delete[] fc2_bias;

    return 0;
}
