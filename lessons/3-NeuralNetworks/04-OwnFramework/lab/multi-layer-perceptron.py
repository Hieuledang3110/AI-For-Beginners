import numpy as np
import pickle
import random
import gzip

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def no_func(x):
    """No activation function."""
    return x

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

# Load the MNIST data with the correct encoding
with gzip.open('mnist.pkl.gz', 'rb') as mnist_pickle:
    MNIST = pickle.load(mnist_pickle, encoding='latin1')  # Specify encoding explicitly

training_data, validation_data, test_data = MNIST  # Unpack the tuple
# Access training features and labels
train_features, train_labels = training_data
val_features, val_labels = validation_data
test_features, test_labels = test_data
# Normalize features to the range [0, 1]
train_features = train_features.astype(np.float32) / 255.0
val_features = val_features.astype(np.float32) / 255.0
test_features = test_features.astype(np.float32) / 255.0

# Multi-layer Perceptron Weights Initialization
def initialize_mlp(layers):
    """
    Initialize weights and biases for a multi-layer perceptron.
    layers: List of layer sizes, e.g., [784, 128, 64, 10]
    Returns:
    - weights: List of weight matrices for each layer
    - biases: List of bias vectors for each layer
    """
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.01)
        biases.append(np.zeros((1, layers[i + 1])))
    return weights, biases

# Forward pass through the network
def forward_mlp(X, weights, biases, activation_fn=sigmoid):
    """
    Perform a forward pass through the MLP.
    X: Input data (batch_size x input_size)
    weights: List of weight matrices
    biases: List of bias vectors
    activation_fn: Activation function to apply
    Returns:
    - activations: List of activations for each layer
    """
    activations = [X]
    for W, b in zip(weights, biases):
        Z = np.dot(activations[-1], W) + b
        A = activation_fn(Z)
        activations.append(A)
    return activations

# Train function for a single layer
def train_layer(X, y, weights, biases, num_iterations, lambda_reg, learning_rate, activation_fn=sigmoid):
    """
    Train a single layer of the MLP.
    X: Input features
    y: Target outputs
    weights, biases: Current weights and biases for the layer
    """
    for i in range(num_iterations):
        idx = random.randint(0, X.shape[0] - 1)
        x_sample = X[idx:idx + 1]  # Random single sample
        y_sample = y[idx:idx + 1]  # Corresponding label

        # Forward pass through the layer
        Z = np.dot(x_sample, weights) + biases
        A = activation_fn(Z)

        # Compute error (difference between prediction and target)
        error = A - y_sample

        # Update weights and biases using the error
        weights -= learning_rate * np.dot(x_sample.T, error) + lambda_reg * weights
        biases -= learning_rate * error

        # # Debug: Print weight updates
        # if i % 10 == 0:  # Every 10 iterations
        #     print(f"Iteration {i}, Weights Norm: {np.linalg.norm(weights):.4f}, Error Norm: {np.linalg.norm(error):.4f}")

    return weights, biases

# Train the entire network
def train_mlp(train_features, train_labels, layers, num_iterations, lambda_reg, learning_rate):
    """
    Train a multi-layer perceptron layer by layer.
    layers: List of layer sizes, e.g., [784, 128, 64, 10]
    """
    weights, biases = initialize_mlp(layers)

    # Train each layer independently
    for l in range(len(layers) - 1):
        print(f"Training layer {l + 1}/{len(layers) - 1}")
        weights[l], biases[l] = train_layer(
            train_features,
            train_labels,
            weights[l],
            biases[l],
            num_iterations,
            lambda_reg,
            learning_rate,
        )
        # Use current layer's output as input to the next layer
        train_features = forward_mlp(train_features, [weights[l]], [biases[l]])[-1]

    return weights, biases

# Classification through the network
def classify_mlp(test_features, weights, biases, activation_fn=relu):
    """
    Classify samples using the trained MLP.
    test_features: Input features
    weights, biases: Trained weights and biases
    """
    activations = forward_mlp(test_features, weights, biases, activation_fn)
    predictions = np.argmax(activations[-1], axis=1)  # Class with highest output
    return predictions

# Define the architecture
layers = [784, 256, 64, 10]  # Example: 2 hidden layers with 128 and 64 neurons

# Train the MLP
weights, biases = train_mlp(
    train_features,
    train_labels,
    layers,
    num_iterations=200,
    lambda_reg=0.01,
    learning_rate=0.1,
)

# Test the MLP
predicted_labels = classify_mlp(test_features, weights, biases)
accuracy = np.mean(predicted_labels == test_labels)
print(f"Test accuracy: {accuracy}")