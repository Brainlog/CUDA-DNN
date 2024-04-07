#include <iostream>
#include <vector>
#include <cmath>
#include <fstream> 

using namespace std;

// Function to perform convolution with padding
vector<vector<float>> convolveWithPadding(const vector<vector<float>>& input, const vector<vector<float>>& kernel) {
    int inputSize = input.size();
    int kernelSize = kernel.size();
    int paddingSize = kernelSize / 2;
    vector<vector<float>> paddedInput(inputSize + 2 * paddingSize, vector<float>(inputSize + 2 * paddingSize, 0.0f));
    vector<vector<float>> output(inputSize, vector<float>(inputSize, 0.0f));

    // Copy input matrix to padded input matrix
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            paddedInput[i + paddingSize][j + paddingSize] = input[i][j];
        }
    }

    // Perform convolution
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            for (int m = 0; m < kernelSize; ++m) {
                for (int n = 0; n < kernelSize; ++n) {
                    output[i][j] += paddedInput[i + m][j + n] * kernel[m][n];
                }
            }
        }
    }

    return output;
}

// Function to perform convolution without padding
vector<vector<float>> convolveWithoutPadding(const vector<vector<float>>& input, const vector<vector<float>>& kernel) {
    int inputSize = input.size();
    int kernelSize = kernel.size();
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


// Function to apply ReLU activation to a matrix
vector<vector<float>> applyReLU(const vector<vector<float>>& input) {
    int rows = input.size();
    int cols = input[0].size();
    vector<vector<float>> output(rows, vector<float>(cols, 0.0f));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[i][j] = max(0.0f, input[i][j]);
        }
    }

    return output;
}

// Function to apply tanh activation to a matrix
vector<vector<float>> applyTanh(const vector<vector<float>>& input) {
    int rows = input.size();
    int cols = input[0].size();
    vector<vector<float>> output(rows, vector<float>(cols, 0.0f));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[i][j] = tanh(input[i][j]);
        }
    }

    return output;
}



// Function to perform max pooling on a square input matrix
vector<vector<float>> maxPooling(const vector<vector<float>>& input, int poolSize) {
    int inputSize = input.size();
    int outputSize = inputSize / poolSize;
    vector<vector<float>> output(outputSize, vector<float>(outputSize, 0.0f));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float maxVal = input[i * poolSize][j * poolSize];
            for (int m = 0; m < poolSize; ++m) {
                for (int n = 0; n < poolSize; ++n) {
                    maxVal = max(maxVal, input[i * poolSize + m][j * poolSize + n]);
                }
            }
            output[i][j] = maxVal;
        }
    }

    return output;
}

// Function to perform average pooling on a square input matrix
vector<vector<float>> avgPooling(const vector<vector<float>>& input, int poolSize) {
    int inputSize = input.size();
    int outputSize = inputSize / poolSize;
    vector<vector<float>> output(outputSize, vector<float>(outputSize, 0.0f));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float sum = 0.0f;
            for (int m = 0; m < poolSize; ++m) {
                for (int n = 0; n < poolSize; ++n) {
                    sum += input[i * poolSize + m][j * poolSize + n];
                }
            }
            output[i][j] = sum / (poolSize * poolSize);
        }
    }

    return output;
}

// Function to apply softmax to a vector
vector<float> softmax(const vector<float>& input) {
    vector<float> output(input.size());
    float sum = 0.0f;
    for (float val : input) {
        sum += exp(val);
    }
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = exp(input[i]) / sum;
    }
    return output;
}

// Function to apply sigmoid to a vector
vector<float> sigmoid(const vector<float>& input) {
    vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = 1.0f / (1.0f + exp(-input[i]));
    }
    return output;
}

// Function to print matrix
void printMatrix(const vector<vector<float>>& matrix) {
    for (const auto& row : matrix) {
        for (float val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

int main(int argc, char** argv) {

    ofstream outFile;
    outFile.open("output_subtask_1.txt");

    if (argc < 2) {
        outFile << "Usage: " << argv[0] << " <mode> [args...]" << endl;
        return 1;
    }

    int mode = atoi(argv[1]);

    if (mode == 1) {  // Convolution
        if (argc < 8) {
            outFile << "Usage: " << argv[0] << " 1 <input_size> <kernel_size> <padding> <input_matrix> <kernel_matrix>" << endl;
            return 1;
        }

        int inputSize = atoi(argv[2]);
        int kernelSize = atoi(argv[3]);
        int padding = atoi(argv[4]);

        vector<vector<float>> input(inputSize, vector<float>(inputSize));
        int argIndex = 5;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                input[i][j] = strtof(argv[argIndex++], nullptr);
            }
        }

        vector<vector<float>> kernel(kernelSize, vector<float>(kernelSize));
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                kernel[i][j] = strtof(argv[argIndex++], nullptr);
            }
        }

        vector<vector<float>> convOutputPadding = convolveWithPadding(input, kernel);
        vector<vector<float>> convOutputNoPadding = convolveWithoutPadding(input, kernel);


        outFile << "Output with padding:" << endl;
        for (const auto& row : convOutputPadding) {
            for (float val : row) {
                outFile << val << " ";
            }
            outFile << endl;
        }

        outFile << "Output without padding:" << endl;
        for (const auto& row : convOutputNoPadding) {
            for (float val : row) {
                outFile << val << " ";
            }
            outFile << endl;
        }

    } else if (mode == 2) {  // Activation
        if (argc < 5) {
            outFile << "Usage: " << argv[0] << " 2 <activation_mode> <matrix_size> <input_matrix>" << endl;
            return 1;
        }

        int activationMode = atoi(argv[2]);
        int matSize = atoi(argv[3]);

        vector<vector<float>> input(matSize, vector<float>(matSize));
        int argIndex = 4;
        for (int i = 0; i < matSize; ++i) {
            for (int j = 0; j < matSize; ++j) {
                input[i][j] = strtof(argv[argIndex++], nullptr);
            }
        }

        vector<vector<float>> output;
        if (activationMode == 0) {
            output = applyReLU(input);
        } else {
            output = applyTanh(input);
        }

        outFile << "Output after activation:" << endl;
        for (const auto& row : output) {
            for (float val : row) {
                outFile << val << " ";
            }
            outFile << endl;
        }

    } else if (mode == 3) {  // Subsampling
        if (argc < 6) {
            outFile << "Usage: " << argv[0] << " 3 <subsampling_mode> <matrix_size> <pool_size> <input_matrix>" << endl;
            return 1;
        }

        int subsamplingMode = atoi(argv[2]);
        int matSize = atoi(argv[3]);
        int poolSize = atoi(argv[4]);

        vector<vector<float>> input(matSize, vector<float>(matSize));
        int argIndex = 5;
        for (int i = 0; i < matSize; ++i) {
            for (int j = 0; j < matSize; ++j) {
                input[i][j] = strtof(argv[argIndex++], nullptr);
            }
        }

        vector<vector<float>> output;
        if (subsamplingMode == 0) {
            output = maxPooling(input, poolSize);
        } else {
            output = avgPooling(input, poolSize);
        }

        outFile << "Output after subsampling:" << endl;
        for (const auto& row : output) {
            for (float val : row) {
                outFile << val << " ";
            }
            outFile << endl;
        }

    } else if (mode == 4) {  // Softmax
        if (argc < 4) {
            outFile << "Usage: " << argv[0] << " 4 <vector_size> <input_vector>" << endl;
            return 1;
        }

        int size = atoi(argv[3]);

        vector<float> input(size);

        int soft_mode = atoi(argv[2]);

        int argIndex = 4;
        for (int i = 0; i < size; ++i) {
            input[i] = strtof(argv[argIndex++], nullptr);
        }
        vector<float> output;
        if(soft_mode == 0){
            output = sigmoid(input);
        }
        else{
            output = softmax(input);
        }        

        outFile << "Output after softmax/sigmoid function:" << endl;
        for (float val : output) {
            outFile << val << " ";
        }
        outFile << endl;

    } else {
        outFile << "Invalid mode" << endl;
    }

    outFile.close();

    return 0;
}