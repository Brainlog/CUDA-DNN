#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Function to perform convolution with padding
vector<vector<float>> convolveWithPadding(float** input, float** kernel, int inputSize, int kernelSize) {
    int paddingSize = kernelSize / 2;
    vector<vector<float>> output(inputSize, vector<float>(inputSize, 0.0f));
    float** paddedInput = new float*[inputSize + 2 * paddingSize];

    // Allocate memory for padded input
    for (int i = 0; i < inputSize + 2 * paddingSize; ++i) {
        paddedInput[i] = new float[inputSize + 2 * paddingSize]();
    }

    // Copy input matrix to padded input matrix
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            paddedInput[i + paddingSize][j + paddingSize] = input[i][j];
        }
    }

    // convolution
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            for (int m = 0; m < kernelSize; ++m) {
                for (int n = 0; n < kernelSize; ++n) {
                    output[i][j] += paddedInput[i + m][j + n] * kernel[m][n];
                }
            }
        }
    }

    // Deallocate memory for padded input
    for (int i = 0; i < inputSize + 2 * paddingSize; ++i) {
        delete[] paddedInput[i];
    }
    delete[] paddedInput;

    return output;
}


vector<vector<float>> convolveWithoutPadding(float** input, float** kernel, int inputSize, int kernelSize) {
    vector<vector<float>> output(inputSize - kernelSize + 1, vector<float>(inputSize - kernelSize + 1, 0.0f));

    // Perform convolution
    for (int i = 0; i < inputSize - kernelSize + 1; ++i) {
        for (int j = 0; j < inputSize - kernelSize + 1; ++j) {
            for (int m = 0; m < kernelSize; ++m) {
                for (int n = 0; n < kernelSize; ++n) {
                    output[i][j] += input[i + m][j + n] * kernel[m][n];
                }
            }
        }
    }

    return output;
}


// Function to apply tanh activation to a matrix
void applyTanh(float** input, int mat_size) { // assuming input is a square matrix rows = cols
    for (int i = 0; i < mat_size; ++i) {
        for (int j = 0; j < mat_size; ++j) {
            input[i][j] = tanh(input[i][j]);
        }
    }
}


// Function to apply ReLU activation to a matrix
void applyReLU(float** input, int mat_size) {
    for (int i = 0; i < mat_size; ++i) {
        for (int j = 0; j < mat_size; ++j) {
            input[i][j] = max(0.0f,input[i][j]);
        }
    }
}


void subsampling(float **matrix, float **final_matrix, bool max_pooling, int mat_size, int final_size, int pool_size)
{
    if (max_pooling)
    {
        float max_element = 0;
        for (int i = 0; i < final_size; i++)
            for (int j = 0; j < final_size; j++)
            {
                max_element = matrix[i * pool_size][j * pool_size];
                for (int k1 = 0; k1 < pool_size; k1++)
                    for (int k2 = 0; k2 < pool_size; k2++)
                        max_element = max(max_element, matrix[i * pool_size + k1][j * pool_size + k2]);
                final_matrix[i][j] = max_element;
            }
    }
    else
    {
        float avg_element = 0;
        for (int i = 0; i < final_size; i++)
            for (int j = 0; j < final_size; j++)
            {
                avg_element = 0;
                for (int k1 = 0; k1 < pool_size; k1++)
                    for (int k2 = 0; k2 < pool_size; k2++)
                        avg_element +=matrix[i * pool_size + k1][j * pool_size + k2];
                final_matrix[i][j] = avg_element/(pool_size*pool_size);
            }
    }
    
}

void softmax(float * vector, float * final_vector,int size)
{
    float denom = 0;
    for(int i=0; i<size; i++)
    denom+=exp(vector[i]);
    for(int i=0; i<size; i++)
    final_vector[i]=exp(vector[i])/denom;
}

void sigmod_function(float * vector, float * final_vector, int size)
{
    for(int i=0; i<size; i++)
    final_vector[i]=1/(1+exp(-vector[i]));
}

int main(int argn, char **argv)
{
    int mat_size, kernel_size;
    if (argn != 3)
    {
        cerr << "not 2 arguments" << endl;
    }
    mat_size = stoi(argv[1]);
    kernel_size = stoi(argv[2]);
    int n_final = mat_size;

    float **conv_matrix = (float **)malloc(n_final * sizeof(float *));
    for (int i = 0; i < n_final; i++)
        conv_matrix[i] = (float *)malloc(n_final * sizeof(float));

    float **matrix = (float **)malloc(mat_size * sizeof(float *));
    for (int i = 0; i < mat_size; i++)
        matrix[i] = (float *)malloc(mat_size * sizeof(float *));

    float **kernel = (float **)malloc(kernel_size * sizeof(float *));
    for (int i = 0; i < kernel_size; i++)
        kernel[i] = (float *)malloc(kernel_size * sizeof(float));

    for (int i = 0; i < mat_size; i++)
        for (int j = 0; j < mat_size; j++)
            matrix[i][j] = int((rand() / ((float)RAND_MAX)) * 10);

    for (int i = 0; i < kernel_size; i++)
        for (int j = 0; j < kernel_size; j++)
            kernel[i][j] = int((rand() / ((float)RAND_MAX)) * 10);

    cout << "Matrix " << endl;
    for (int i = 0; i < mat_size; i++)
    {
        for (int j = 0; j < mat_size; j++)
            cout << matrix[i][j] << " ";
        cout << endl;
    }

    cout << "Kernel" << endl;
    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
            cout << kernel[i][j] << " ";
        cout << endl;
    }

    conv(matrix, kernel, conv_matrix, true, mat_size, kernel_size);

    cout << "convolved matrix " << endl;
    for (int i = 0; i < n_final; i++)
    {
        for (int j = 0; j < n_final; j++)
            cout << conv_matrix[i][j] << " ";
        cout << endl;
    }
}
