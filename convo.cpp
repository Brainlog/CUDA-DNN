#include <iostream>
#include <vector>
#include <cmath>

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


int main() {
    // Define input matrix
    vector<vector<float>> input = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f},
        {9.0f, 10.0f, 11.0f, 12.0f},
        {13.0f, 14.0f, 15.0f, 16.0f}
    };

    // Define kernel matrix
    vector<vector<float>> kernel = {
        {1.0f, 0.0f, -1.0f},
        {1.0f, 0.0f, -1.0f},
        {1.0f, 0.0f, -1.0f}
    };

    // Perform convolution with padding
    cout << "Convolution with padding:" << endl;
    vector<vector<float>> convOutputPadding = convolveWithPadding(input, kernel);
    printMatrix(convOutputPadding);
    cout << endl;

    // Perform convolution without padding
    cout << "Convolution without padding:" << endl;
    vector<vector<float>> convOutputNoPadding = convolveWithoutPadding(input, kernel);
    printMatrix(convOutputNoPadding);
    cout << endl;

    // Apply ReLU activation
    cout << "ReLU Activation:" << endl;
    vector<vector<float>> reluOutput = applyReLU(input);
    printMatrix(reluOutput);
    cout << endl;

    // Apply tanh activation
    cout << "Tanh Activation:" << endl;
    vector<vector<float>> tanhOutput = applyTanh(input);
    printMatrix(tanhOutput);
    cout << endl;

    // Perform max pooling
    cout << "Max Pooling:" << endl;
    vector<vector<float>> maxPoolOutput = maxPooling(input, 2);
    printMatrix(maxPoolOutput);
    cout << endl;

    // Perform average pooling
    cout << "Average Pooling:" << endl;
    vector<vector<float>> avgPoolOutput = avgPooling(input, 2);
    printMatrix(avgPoolOutput);
    cout << endl;

    // Apply softmax to a vector
    cout << "Softmax Activation:" << endl;
    vector<float> softmaxOutput = softmax({1.0f, 2.0f, 3.0f});
    for (float val : softmaxOutput) {
        cout << val << " ";
    }
    cout << endl << endl;

    // Apply sigmoid to a vector
    cout << "Sigmoid Activation:" << endl;
    vector<float> sigmoidOutput = sigmoid({1.0f, 2.0f, 3.0f});
    for (float val : sigmoidOutput) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
