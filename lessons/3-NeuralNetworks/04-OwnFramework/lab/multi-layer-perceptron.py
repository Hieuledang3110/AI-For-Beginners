import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import random
import gzip

# Load the MNIST data with the correct encoding
with gzip.open('mnist.pkl.gz', 'rb') as mnist_pickle:
    MNIST = pickle.load(mnist_pickle, encoding='latin1')  # Specify encoding explicitly

training_data, validation_data, test_data = MNIST  # Unpack the tuple
# Access training features and labels
train_features, train_labels = training_data
val_features, val_labels = validation_data
test_features, test_labels = test_data
# Normalize features to the range [0, 1] to improve training stability
# Note: This is a common practice for many machine learning algorithm. MNIST dataset contains grayscale images, where each pixel value ranges from 0 to 255
train_features = train_features.astype(np.float32) / 255.0
val_features = val_features.astype(np.float32) / 255.0
test_features = test_features.astype(np.float32) / 255.0

# Filter positive and negative examples for training
def set_mnist_pos_neg(target_label, x_labels, x_features):
    positive_indices = [i for i, label in enumerate(x_labels) if label == target_label]
    negative_indices = [i for i, label in enumerate(x_labels) if label != target_label]

    positive_images = x_features[positive_indices]
    negative_images = x_features[negative_indices]
    
    return positive_images, negative_images

# Train function for a single binary classifier (one-vs-all)
def train(positive_examples, negative_examples, num_iterations, lambda_reg, weights):
    num_dims = positive_examples.shape[1]  # Number of features
    if weights is None:  # Initialize weights if not provided
        weights = np.zeros((num_dims, 1))*0.01 # Shape: (num_features, 1), initialized with small values to prevent convergence issues

    for i in range(num_iterations):  # Optimize weights through gradient descent
        pos = random.choice(positive_examples).reshape(-1, 1)  # Shape: (num_features, 1)
        neg = random.choice(negative_examples).reshape(-1, 1)  # Shape: (num_features, 1)

        # Update weights based on positive and negative examples
        if np.dot(weights.T, pos) < 0:
            weights += pos
        if np.dot(weights.T, neg) >= 0:
            weights -= neg
        # Apply L2 regularization
        weights -= lambda_reg * weights  # Regularization step

    return weights

# Train one-vs-all classifiers for one layer with weight updates
def train_all_classes(train_features, train_labels, num_iterations, lambda_reg, weights_list):
    if weights_list is None:
        # Initialize weights for 10 classes if not provided
        weights_list = [np.zeros((train_features.shape[1], 1)) for _ in range(10)]  

    updated_weights_list = []
    for digit in range(10):  # Train one classifier per digit (0-9)
        pos_examples, neg_examples = set_mnist_pos_neg(digit, train_labels, train_features)
        # Pass existing weights to the train function to update them
        weights = train(pos_examples, neg_examples, num_iterations, lambda_reg, weights_list[digit])
        updated_weights_list.append(weights)

    return updated_weights_list


# Multi-layer perceptron with weight updating across layers
def multi_layer_training(train_features, train_labels, num_iterations, lambda_reg, num_layers):
    weights_list = None  # Initialize weights list for the first layer
    accuracies = []  # Track accuracies for each layer
    for layer in range(num_layers):
        print(f"Training Layer {layer + 1}")
        weights_list = train_all_classes(train_features, train_labels, num_iterations, lambda_reg, weights_list)
        # Evaluate the current weights on the test set
        predicted_labels = classify_multi_class(weights_list, test_features)
        layer_accuracy = accuracy(predicted_labels, test_labels)
        accuracies.append(layer_accuracy)
        print(f"Accuracy after Layer {layer + 1}: {layer_accuracy:.4f}")
    return weights_list, accuracies


# Classification Function with Matrix Multiplication
def classify_multi_class(weights_list, test_features):
    # Convert weights list to a (num_features, 10) matrix
    weights_matrix = np.column_stack(weights_list)  # Shape: (num_features, 10)
    # Compute scores: test_features (num_samples, num_features) * weights_matrix (num_features, 10)
    scores = np.dot(test_features, weights_matrix)  # Shape: (num_samples, 10)
    # For each test sample, pick the class with the highest score
    predicted_labels = np.argmax(scores, axis=1)  # Shape: (num_samples, ) - max score index for each sample
    return predicted_labels

def accuracy(predicted_labels, test_labels):
    return float(np.sum(predicted_labels == test_labels) / len(test_labels))

# Train and evaluate the multi-layer perceptron
num_layers = 3  # Define the number of layers
weights_list, accuracies = multi_layer_training(train_features, train_labels, 200, -0.015, num_layers)
# Final accuracy after all layers
print("Final multi-layer perceptron accuracy:", accuracies[-1])