import numpy as np

# Define the Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)  # Random weight initialization
        self.bias = np.random.rand(1)              # Random bias

    def forward(self, inputs):
        total_input = np.dot(inputs, self.weights) + self.bias
        return sigmoid(total_input)

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                output = self.forward(X[i])
                error = y[i] - output
                # Update weights and bias using gradient descent
                self.weights += learning_rate * error * X[i]
                self.bias += learning_rate * error

# Dataset for AND and OR functions
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])

# Initialize perceptrons
perceptron_and = Perceptron(input_size=2)
perceptron_or = Perceptron(input_size=2)

# Train perceptrons
perceptron_and.train(X, y_and, epochs=1000, learning_rate=0.1)
perceptron_or.train(X, y_or, epochs=1000, learning_rate=0.1)

# Test perceptrons
print("AND Function Predictions:")
for i in range(X.shape[0]):
    output = perceptron_and.forward(X[i])
    print(f"Input: {X[i]} - Predicted Output: {round(float(output))}")

print("\nOR Function Predictions:")
for i in range(X.shape[0]):
    output = perceptron_or.forward(X[i])
    print(f"Input: {X[i]} - Predicted Output: {round(float(output))}")
