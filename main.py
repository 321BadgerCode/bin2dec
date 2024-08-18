import os
import tensorflow as tf
import numpy as np
import random

class BinaryToDecimalNN:
	def __init__(self, input_size):
		self.input_size = input_size
		self.model = self._build_model()

	def _build_model(self):
		model = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=(self.input_size,)),
			tf.keras.layers.Dense(64, activation='relu'),
			tf.keras.layers.Dense(32, activation='relu'),
			tf.keras.layers.Dense(1, activation='linear')
		])
		model.compile(optimizer='adam', loss='mean_squared_error')
		return model

	def train(self, binary_inputs, decimal_outputs, epochs=100):
		x_train = np.array([self._binary_to_one_hot(bi) for bi in binary_inputs])
		y_train = np.array(decimal_outputs)
		self.model.fit(x_train, y_train, epochs=epochs, verbose=1)

	def predict(self, binary_input):
		x_test = np.array([self._binary_to_one_hot(binary_input)])
		y_pred = self.model.predict(x_test)
		return int(round(y_pred[0][0]))

	def _binary_to_one_hot(self, binary_input):
		return np.array([int(bit) for bit in binary_input] + [0] * (self.input_size - len(binary_input)))

	def save(self, filename):
		self.model.save(filename)

	def load(self, filename):
		self.model = tf.keras.models.load_model(filename)

def generate_binary_data(min_length, max_length, num_samples):
	binary_inputs = []
	decimal_outputs = []

	for _ in range(num_samples):
		length = random.randint(min_length, max_length)
		binary_str = ''.join(random.choice('01') for _ in range(length))
		decimal_value = int(binary_str, 2)
		binary_str_padded = binary_str.zfill(max_length)
		binary_inputs.append(binary_str_padded)
		decimal_outputs.append(decimal_value)
	
	return binary_inputs, decimal_outputs

def main():
	min_length = 3
	max_length = 9
	num_samples = 1000 # Number of binary samples to generate
	input_size = max_length # Set input size to the maximum length of the binary strings

	nn = BinaryToDecimalNN(input_size)

	if not os.path.exists("./model/"):
		# Train the model
		binary_inputs, decimal_outputs = generate_binary_data(min_length, max_length, num_samples)
		nn.train(binary_inputs, decimal_outputs, epochs=200)
		nn.save("./model/")
	nn.load("./model/")

	# Test the model with new binary inputs
	test_inputs = generate_binary_data(min_length, max_length, 10)[0]
	for binary_input in test_inputs:
		decimal_output = nn.predict(binary_input.zfill(input_size))
		print(f'Binary: {binary_input}, Decimal: {decimal_output}')

if __name__ == "__main__":
	main()