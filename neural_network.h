#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>

using namespace std;

class NeuralNetwork {
public:
	NeuralNetwork(int inputSize, int hiddenSize, int outputSize);

	void train(const vector<vector<int>>& binaryInputs,
			   const vector<int>& decimalOutputs,
			   int epochs, double learningRate);

	int predict(const vector<int>& binaryInput);

	void saveToFile(const string& filename);
	void loadFromFile(const string& filename);

private:
	int inputSize;
	int hiddenSize;
	int outputSize;
	vector<vector<double>> weightsInputHidden;
	vector<vector<double>> weightsHiddenOutput;
	vector<double> biasHidden;
	vector<double> biasOutput;

	vector<double> sigmoid(const vector<double>& z);
	vector<double> sigmoidDerivative(const vector<double>& z);
	double meanSquaredError(const vector<int>& outputs, const vector<double>& predictions);
	void forwardPropagate(const vector<int>& input, vector<double>& hiddenLayer, vector<double>& outputLayer);
	void backPropagate(const vector<int>& input, const vector<double>& hiddenLayer, const vector<double>& outputLayer, double learningRate, int expectedOutput);
	vector<int> binaryToOneHot(const vector<int>& binaryInput);
};

#endif // NEURALNETWORK_H