#include "neural_network.h"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>

using namespace std;

// Helper function to initialize weights randomly
vector<vector<double>> initializeWeights(int rows, int cols) {
	vector<vector<double>> weights(rows, vector<double>(cols));
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // Random values between -1 and 1
		}
	}
	return weights;
}

// Neural Network Constructor
NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
	: inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) {
	srand(time(0));
	weightsInputHidden = initializeWeights(inputSize, hiddenSize);
	weightsHiddenOutput = initializeWeights(hiddenSize, outputSize);
	biasHidden = vector<double>(hiddenSize, 0.0);
	biasOutput = vector<double>(outputSize, 0.0);
}

vector<double> NeuralNetwork::sigmoid(const vector<double>& z) {
	vector<double> result(z.size());
	for (size_t i = 0; i < z.size(); ++i) {
		result[i] = 1.0 / (1.0 + exp(-z[i]));
	}
	return result;
}

vector<double> NeuralNetwork::sigmoidDerivative(const vector<double>& z) {
	vector<double> sigmoidValues = sigmoid(z);
	vector<double> result(z.size());
	for (size_t i = 0; i < sigmoidValues.size(); ++i) {
		result[i] = sigmoidValues[i] * (1 - sigmoidValues[i]);
	}
	return result;
}

double NeuralNetwork::meanSquaredError(const vector<int>& outputs, const vector<double>& predictions) {
	double error = 0.0;
	for (size_t i = 0; i < outputs.size(); ++i) {
		double diff = outputs[i] - predictions[i];
		error += diff * diff;
	}
	return error / outputs.size();
}

void NeuralNetwork::forwardPropagate(const vector<int>& input, vector<double>& hiddenLayer, vector<double>& outputLayer) {
	// Compute hidden layer
	for (int j = 0; j < hiddenSize; ++j) {
		double sum = 0.0;
		for (int i = 0; i < inputSize; ++i) {
			sum += input[i] * weightsInputHidden[i][j];
		}
		sum += biasHidden[j];
		hiddenLayer[j] = 1.0 / (1.0 + exp(-sum));
	}

	// Compute output layer
	for (int k = 0; k < outputSize; ++k) {
		double sum = 0.0;
		for (int j = 0; j < hiddenSize; ++j) {
			sum += hiddenLayer[j] * weightsHiddenOutput[j][k];
		}
		sum += biasOutput[k];
		outputLayer[k] = sum; // Linear activation for regression
	}
}

void NeuralNetwork::backPropagate(const vector<int>& input, const vector<double>& hiddenLayer, const vector<double>& outputLayer, double learningRate, int expectedOutput) {
	vector<double> outputLayerError(outputSize);
	for (int k = 0; k < outputSize; ++k) {
		outputLayerError[k] = (expectedOutput - outputLayer[k]);
	}

	// Update weights and biases for output layer
	for (int k = 0; k < outputSize; ++k) {
		for (int j = 0; j < hiddenSize; ++j) {
			weightsHiddenOutput[j][k] += learningRate * outputLayerError[k] * hiddenLayer[j];
		}
		biasOutput[k] += learningRate * outputLayerError[k];
	}

	// Update weights and biases for hidden layer
	vector<double> hiddenLayerError(hiddenSize);
	for (int j = 0; j < hiddenSize; ++j) {
		double sum = 0.0;
		for (int k = 0; k < outputSize; ++k) {
			sum += outputLayerError[k] * weightsHiddenOutput[j][k];
		}
		hiddenLayerError[j] = sum * hiddenLayer[j] * (1 - hiddenLayer[j]);
	}

	for (int j = 0; j < hiddenSize; ++j) {
		for (int i = 0; i < inputSize; ++i) {
			weightsInputHidden[i][j] += learningRate * hiddenLayerError[j] * input[i];
		}
		biasHidden[j] += learningRate * hiddenLayerError[j];
	}
}

vector<int> NeuralNetwork::binaryToOneHot(const vector<int>& binaryInput) {
	vector<int> oneHotInput(inputSize, 0);
	for (size_t i = 0; i < binaryInput.size(); ++i) {
		oneHotInput[i] = binaryInput[i];
	}
	return oneHotInput;
}

void NeuralNetwork::train(const vector<vector<int>>& binaryInputs,
						  const vector<int>& decimalOutputs,
						  int epochs, double learningRate) {
	for (int epoch = 0; epoch < epochs; ++epoch) {
		double totalError = 0.0;
		for (size_t i = 0; i < binaryInputs.size(); ++i) {
			vector<double> hiddenLayer(hiddenSize);
			vector<double> outputLayer(outputSize);

			forwardPropagate(binaryInputs[i], hiddenLayer, outputLayer);

			double error = meanSquaredError({decimalOutputs[i]}, outputLayer);
			totalError += error;

			backPropagate(binaryInputs[i], hiddenLayer, outputLayer, learningRate, decimalOutputs[i]);
		}
		if (epoch % 10 == 0) {
			cout << "Epoch " << epoch << ": Error = " << totalError / binaryInputs.size() << endl;
		}
	}
}

int NeuralNetwork::predict(const vector<int>& binaryInput) {
	vector<double> hiddenLayer(hiddenSize);
	vector<double> outputLayer(outputSize);

	forwardPropagate(binaryInput, hiddenLayer, outputLayer);

	return static_cast<int>(round(outputLayer[0])); // Round to nearest integer
}

void NeuralNetwork::saveToFile(const string& filename) {
	ofstream outFile(filename, ios::binary);
	if (!outFile) {
		cerr << "Error: Could not open file " << filename << " for writing." << endl;
		return;
	}

	// Save input size, hidden size, output size
	outFile.write(reinterpret_cast<const char*>(&inputSize), sizeof(inputSize));
	outFile.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(hiddenSize));
	outFile.write(reinterpret_cast<const char*>(&outputSize), sizeof(outputSize));

	// Save weights from input to hidden layer
	for (const auto& row : weightsInputHidden) {
		outFile.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
	}

	// Save weights from hidden to output layer
	for (const auto& row : weightsHiddenOutput) {
		outFile.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
	}

	// Save biases for hidden layer
	outFile.write(reinterpret_cast<const char*>(biasHidden.data()), biasHidden.size() * sizeof(double));

	// Save biases for output layer
	outFile.write(reinterpret_cast<const char*>(biasOutput.data()), biasOutput.size() * sizeof(double));

	outFile.close();
}

void NeuralNetwork::loadFromFile(const string& filename) {
	ifstream inFile(filename, ios::binary);
	if (!inFile) {
		cerr << "Error: Could not open file " << filename << " for reading." << endl;
		return;
	}

	// Load input size, hidden size, output size
	inFile.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
	inFile.read(reinterpret_cast<char*>(&hiddenSize), sizeof(hiddenSize));
	inFile.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));

	// Resize vectors to match loaded dimensions
	weightsInputHidden.resize(inputSize, vector<double>(hiddenSize));
	weightsHiddenOutput.resize(hiddenSize, vector<double>(outputSize));
	biasHidden.resize(hiddenSize);
	biasOutput.resize(outputSize);

	// Load weights from input to hidden layer
	for (auto& row : weightsInputHidden) {
		inFile.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
	}

	// Load weights from hidden to output layer
	for (auto& row : weightsHiddenOutput) {
		inFile.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
	}

	// Load biases for hidden layer
	inFile.read(reinterpret_cast<char*>(biasHidden.data()), biasHidden.size() * sizeof(double));

	// Load biases for output layer
	inFile.read(reinterpret_cast<char*>(biasOutput.data()), biasOutput.size() * sizeof(double));

	inFile.close();
}