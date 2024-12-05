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
def train(positive_examples, negative_examples, num_iterations, lambda_reg):
    num_dims = positive_examples.shape[1]  # Adjust for bias term, no need for extra feature column
    weights = np.zeros((num_dims, 1))  # Initialize weights (without bias term yet)

    for i in range(num_iterations): # Optimize weights through gradient descent
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)
        if np.dot(pos, weights) < 0:
            weights += pos.reshape(weights.shape)
        if np.dot(neg, weights) >= 0:
            weights -= neg.reshape(weights.shape)
        # L2 Regularization: Shrink weights slightly to prevent large weights
        weights -= lambda_reg * weights  # Regularization step

    return weights

# Train one-vs-all classifiers
def train_all_classes(train_features, train_labels, num_iterations, lambda_reg):
    weights_list = []
    for digit in range(10):  # Train one classifier per digit (0-9)
        pos_examples, neg_examples = set_mnist_pos_neg(digit, train_labels, train_features)
        weights = train(pos_examples, neg_examples, num_iterations, lambda_reg)
        weights_list.append(weights)
    return weights_list

# Classification Function with Matrix Multiplication
def classify_multi_class(weights_list, test_features, test_labels):
    # Convert weights list to a (num_features, 10) matrix
    weights_matrix = np.column_stack(weights_list)  # Shape: (num_features, 10)

    # Compute scores: test_features (num_samples, num_features) * weights_matrix (num_features, 10)
    scores = np.dot(test_features, weights_matrix)  # Shape: (num_samples, 10)

    # For each test sample, pick the class with the highest score
    predicted_labels = np.argmax(scores, axis=1)  # Shape: (num_samples, ) - max score index for each sample

    return predicted_labels

def accuracy(predicted_labels, test_labels):
    return float(np.sum(predicted_labels == test_labels) / len(test_labels))

# Running multi-class experiments and finding the best lambda_reg
def run_experiment_multi_class():
    best_lambda = None
    best_accuracy = 0

    # Define a range of values for lambda_reg and num_iterations
    lambda_values = np.arange(start=-0.2, stop=0.2, step=0.005)
    num_runs = 10
    
    # Store results
    results = {}
    for lambda_reg in lambda_values:
        # Store accuracies for multiple runs
        accuracies = []
        for run in range(num_runs):
            # Train the multi-class model for current lambda_reg
            weights_list = train_all_classes(train_features, train_labels, num_iterations=200, lambda_reg=lambda_reg)
            # Evaluate accuracy on the validation set
            val_acc = accuracy(classify_multi_class(weights_list, test_features, test_labels), test_labels)
            accuracies.append(val_acc)
        
        # Calculate the average accuracy for this combination of lambda_reg and num_iterations
        avg_accuracy = np.mean(accuracies)
        results[(lambda_reg)] = avg_accuracy
        print(f"Average accuracy for lambda_reg={lambda_reg}: {avg_accuracy:.4f}")
        
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_lambda = lambda_reg
    
    # Output the best parameters
    print(f"\nBest lambda_reg: {best_lambda}, Best accuracy: {best_accuracy:.4f}")
    
    return results

# Run the multi-class experiments
# results = run_experiment_multi_class()
weights_list = train_all_classes(train_features, train_labels, 200, -0.015)
predicted_labels = classify_multi_class(weights_list, test_features, test_labels)
print("Training one-vs-all classifiers accuracy: " + str(accuracy(predicted_labels, test_labels)))