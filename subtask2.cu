#include <bits/stdc++.h>
using namespace std;

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
            out[outchannel  + (row * outsize + col)*(kchannels*inchannels)] = sum;
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

