#include "neural_network.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>

using namespace std;

#define VERSION "1.0.0"

unsigned int lfsr(unsigned int seed) {
	// Tap positions for a 32-bit LFSR (feedback polynomial: x^32 + x^22 + x^2 + x^1 + 1)
	unsigned int lsb = seed & 1; // Get the least significant bit
	seed >>= 1; // Shift the seed right by 1 bit
	if (lsb) { // If lsb is 1, apply the feedback polynomial
		seed ^= 0x80000057; // Apply tap positions for feedback
	}
	return seed;
}

void generateBinaryData(int minLength, int maxLength, int numSamples,
			vector<vector<int>>& binaryInputs,
			vector<int>& decimalOutputs) {
	binaryInputs.clear();
	decimalOutputs.clear();

	unsigned int seed = static_cast<unsigned int>(time(0)); // Use current time as seed

	for (int i = 0; i < numSamples; ++i) {
		seed = lfsr(seed); // Update seed using LFSR
		int length = minLength + (seed % (maxLength - minLength + 1)); // Random length between minLength and maxLength

		vector<int> binaryString(length);
		for (int j = 0; j < length; ++j) {
			seed = lfsr(seed); // Update seed for each bit
			binaryString[j] = seed & 1; // Get the least significant bit as the next binary digit
		}

		// Convert binary string to decimal value
		int decimalValue = 0;
		for (int bit : binaryString) {
			decimalValue = (decimalValue << 1) | bit; // Shift left and add bit
		}

		// Pad binary string to maxLength
		while (binaryString.size() < maxLength) {
			binaryString.insert(binaryString.begin(), 0); // Insert leading zeros
		}

		binaryInputs.push_back(binaryString);
		decimalOutputs.push_back(decimalValue);
	}
}

vector<int> binaryStringToVector(const string& binaryString, const int length) {
	vector<int> binaryVector;
	for (char bit : binaryString) {
		binaryVector.push_back(bit - '0');
	}
	while (binaryVector.size() < length) {
		binaryVector.insert(binaryVector.begin(), 0);
	}
	return binaryVector;
}

int main(int argc, char** argv) {
	// Get command line arguments
	string binaryString = "";
	for (int i = 1; i < argc; i++) {
		if (string(argv[i]) == "-h" || string(argv[i]) == "--help") {
			cout << "Usage: " << argv[0] << " [options]" << endl;
			cout << "Options:" << endl;
			cout << "-h, --help\t\tShow this help message" << endl;
			cout << "--version\t\tShow version information" << endl;
			cout << "[binary string]\t\tBinary string to predict decimal value" << endl;
			return 0;
		} else if (string(argv[i]) == "--version") {
			cout << "Version: " << VERSION << endl;
			return 0;
		} else {
			binaryString = argv[i];
		}
	}

	int inputSize = 10; // Example max length for binary inputs
	int hiddenSize = 50; // Size of hidden layer
	int outputSize = 1; // Size of output layer

	NeuralNetwork nn(inputSize, hiddenSize, outputSize);

	int minLength = 3;
	int maxLength = inputSize;
	int numSamples = 10000; // Number of binary samples to generate

	vector<vector<int>> binaryInputs;
	vector<int> decimalOutputs;

	// Train if model doesn't exist
	if (ifstream("./model.bin")) {
		nn.loadFromFile("./model.bin");
	} else {
		// Generate synthetic training data
		generateBinaryData(minLength, maxLength, numSamples, binaryInputs, decimalOutputs);

		// Train the model
		nn.train(binaryInputs, decimalOutputs, 100, 0.01);
		nn.saveToFile("model.bin");
	}

	// Test the model
	if (binaryString != "") {
		vector<int> binaryVector = binaryStringToVector(binaryString, inputSize);
		int prediction = nn.predict(binaryVector);
		cout << "Prediction: " << prediction << endl;
	} else {
		int testNumSamples = 100;
		vector<vector<int>> testInputs;
		vector<int> testOutputs;
		generateBinaryData(minLength, maxLength, testNumSamples, testInputs, testOutputs);
		int correct = 0;
		for (int i = 0; i < testNumSamples; i++) {
			int prediction = nn.predict(testInputs[i]);
			if (prediction == testOutputs[i]) {
				correct++;
			}
		}
		cout << "Accuracy: " << ((double)correct / testNumSamples) * 100 << "%" << endl;
	}

	return 0;
}