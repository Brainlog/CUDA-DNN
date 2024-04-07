#include <iostream>
#include <vector>
#include <cmath>
#include <fstream> 

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



int main(int argc, char** argv) {

    ofstream outFile;
    outFile.open("output_subtask_1.txt");

    if (argc < 2) {
        outFile << "Usage: " << argv[0] << " <mode> [args...]" << endl;
        outFile.close();
        return 1;
    }

    int mode = atoi(argv[1]);

    if (mode == 1) {  // Convolution
        if (argc < 7) {
            outFile << "Usage: " << argv[0] << " 1 <input_size> <kernel_size> <input_matrix> <kernel_matrix>" << endl;
            outFile.close();
            return 1;
        }

        int inputSize = atoi(argv[2]);
        int kernelSize = atoi(argv[3]);

        // Allocate memory for input and kernel matrices
        float** input = new float*[inputSize];
        for (int i = 0; i < inputSize; ++i) {
            input[i] = new float[inputSize];
        }
        float** kernel = new float*[kernelSize];
        for (int i = 0; i < kernelSize; ++i) {
            kernel[i] = new float[kernelSize];
        }

        // Initialize input matrix
        int argIndex = 4;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                input[i][j] = atof(argv[argIndex++]);
            }
        }

        // Initialize kernel matrix
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                kernel[i][j] = atof(argv[argIndex++]);
            }
        }

        // Perform convolution with padding
        vector<vector<float>> convOutputPadding = convolveWithPadding(input, kernel, inputSize, kernelSize);
        outFile << "Output with padding:" << endl;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                outFile << convOutputPadding[i][j] << " ";
            }
            outFile << endl;
        }

        // Deallocate memory for input and kernel matrices
        for (int i = 0; i < inputSize; ++i) {
            delete[] input[i];
        }
        delete[] input;
        for (int i = 0; i < kernelSize; ++i) {
            delete[] kernel[i];
        }
        delete[] kernel;

    } else if (mode == 2) {  // Activation
        if (argc < 5) {
            outFile << "Usage: " << argv[0] << " 2 <activation_mode> <matrix_size> <input_matrix>" << endl;
            outFile.close();
            return 1;
        }

        int activationMode = atoi(argv[2]);
        int matSize = atoi(argv[3]);

        // Allocate memory for input matrix
        float** input = new float*[matSize];
        for (int i = 0; i < matSize; ++i) {
            input[i] = new float[matSize];
        }

        // Initialize input matrix
        int argIndex = 4;
        for (int i = 0; i < matSize; ++i) {
            for (int j = 0; j < matSize; ++j) {
                input[i][j] = atof(argv[argIndex++]);
            }
        }

        // Apply activation function
        if (activationMode == 0) {
            applyReLU(input, matSize);
        } else {
            applyTanh(input, matSize);
        }

        // Print output matrix
        outFile << "Output after activation:" << endl;
        for (int i = 0; i < matSize; ++i) {
            for (int j = 0; j < matSize; ++j) {
                outFile << input[i][j] << " ";
            }
            outFile << endl;
        }

        // Deallocate memory for input matrix
        for (int i = 0; i < matSize; ++i) {
            delete[] input[i];
        }
        delete[] input;

    } else if (mode == 3) {  // Subsampling
        if (argc < 6) {
            outFile << "Usage: " << argv[0] << " 3 <subsampling_mode> <matrix_size> <final_size> <pool_size> <input_matrix>" << endl;
            outFile.close();
            return 1;
        }

        int subsamplingMode = atoi(argv[2]);
        int matSize = atoi(argv[3]);
        int finalSize = atoi(argv[4]);
        int poolSize = atoi(argv[5]);

        // Allocate memory for input and final matrices
        float** input = new float*[matSize];
        for (int i = 0; i < matSize; ++i) {
            input[i] = new float[matSize];
        }
        float** finalMatrix = new float*[finalSize];
        for (int i = 0; i < finalSize; ++i) {
            finalMatrix[i] = new float[finalSize];
        }

        // Initialize input matrix
        int argIndex = 6;
        for (int i = 0; i < matSize; ++i) {
            for (int j = 0; j < matSize; ++j) {
                input[i][j] = atof(argv[argIndex++]);
            }
        }

        // Perform subsampling
        subsampling(input, finalMatrix, subsamplingMode == 0, matSize, finalSize, poolSize);

        // Print output matrix
        outFile << "Output after subsampling:" << endl;
        for (int i = 0; i < finalSize; ++i) {
            for (int j = 0; j < finalSize; ++j) {
                outFile << finalMatrix[i][j] << " ";
            }
            outFile << endl;
        }

        // Deallocate memory for input and final matrices
        for (int i = 0; i < matSize; ++i) {
            delete[] input[i];
        }
        delete[] input;
        for (int i = 0; i < finalSize; ++i) {
            delete[] finalMatrix[i];
        }
        delete[] finalMatrix;

    } else if (mode == 4) {  // Softmax
        if (argc < 3) {
            outFile << "Usage: " << argv[0] << " 4 <vector_size> <input_vector>" << endl;
            outFile.close();
            return 1;
        }

        int size = atoi(argv[2]);

        // Allocate memory for input vector
        float* input = new float[size];

        // Initialize input vector
        int argIndex = 3;
        for (int i = 0; i < size; ++i) {
            input[i] = atof(argv[argIndex++]);
        }

        // Allocate memory for output vector
        float* output = new float[size];

        // Apply softmax function
        softmax(input, output, size);

        // Print output vector
        outFile << "Output after softmax function:" << endl;
        for (int i = 0; i < size; ++i) {
            outFile << output[i] << " ";
        }
        outFile << endl;

        // Deallocate memory for input and output vectors
        delete[] input;
        delete[] output;

    } else if (mode == 5) {  // Sigmoid
        if (argc < 3) {
            outFile << "Usage: " << argv[0] << " 5 <vector_size> <input_vector>" << endl;
                        outFile.close();
            return 1;
        }

        int size = atoi(argv[2]);

        // Allocate memory for input vector
        float* input = new float[size];

        // Initialize input vector
        int argIndex = 3;
        for (int i = 0; i < size; ++i) {
            input[i] = atof(argv[argIndex++]);
        }

        // Allocate memory for output vector
        float* output = new float[size];

        // Apply sigmoid function
        sigmod_function(input, output, size);

        // Print output vector
        outFile << "Output after sigmoid function:" << endl;
        for (int i = 0; i < size; ++i) {
            outFile << output[i] << " ";
        }
        outFile << endl;

        // Deallocate memory for input and output vectors
        delete[] input;
        delete[] output;

    } else {
        outFile << "Invalid mode" << endl;
    }

    outFile.close();

    return 0;
}

