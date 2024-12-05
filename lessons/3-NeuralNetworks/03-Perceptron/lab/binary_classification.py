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
# Debugging information: features of different datasets (sample size, # features)
# print("Train features shape:", train_features.shape)
# print("Validation features shape:", val_features.shape)
# print("Test features shape:", test_features.shape)

# Filter positive and negative examples for training
def set_mnist_pos_neg(positive_label, negative_label, x_labels, x_features):
    positive_indices = [i for i, label in enumerate(x_labels) if label == positive_label]
    negative_indices = [i for i, label in enumerate(x_labels) if label == negative_label]

    positive_images = x_features[positive_indices]
    negative_images = x_features[negative_indices]

    # print("Positive indices count:", len(positive_indices))
    # print("Negative indices count:", len(negative_indices))
    
    return positive_images, negative_images


def train(positive_examples, negative_examples, num_iterations, lambda_reg):
    num_dims = positive_examples.shape[1]  # Adjust for bias term, no need for extra feature column
    weights = np.zeros((num_dims,1)) # Initialize weights (without bias term yet)
    
    # report_frequency = 50
    
    for i in range(num_iterations): # Optimize weights through gradient descent
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)

        z = np.dot(pos, weights)   
        if z < 0:
            weights += pos.reshape(weights.shape)

        z  = np.dot(neg, weights)
        if z >= 0:
            weights -= neg.reshape(weights.shape)

        # L2 Regularization: Shrink weights slightly to prevent large weights
        # Higher values of lambda_reg will shrink the weights more, reducing the risk of overfitting
        weights -= lambda_reg * weights  # Regularization step
            
        # if i % report_frequency == 0:             
        #     pos_out = np.dot(positive_examples, weights)
        #     neg_out = np.dot(negative_examples, weights)        
        #     pos_correct = (pos_out >= 0).sum() / float(len(positive_examples))
        #     neg_correct = (neg_out < 0).sum() / float(len(negative_examples))
        #     print(f"Iteration={i}, pos correct={pos_correct:.2f}, neg correct={neg_correct:.2f}")

    return weights



def accuracy(weights, test_features, test_labels):
    # Debugging information: test features (sample size, # features), weights
    # print("Test features shape:", test_features.shape)
    # print("Weights shape:", weights.shape)
    
    # Compute predictions
    res = np.dot(test_features, weights)  # Perform dot product with weights (including bias)
    return (res.reshape(test_labels.shape) * test_labels >= 0).sum() / float(len(test_labels))


# Function to run experiments over different lambda_reg and num_iterations
def run_experiments_using_validation():
    best_lambda = None
    best_accuracy = 0
    
    # Define a range of values for lambda_reg and num_iterations
    lambda_values = np.arange(start=-0.2, stop=0.2, step=0.005)
    num_runs = 20
    
    # Store results
    results = {}

    # Filter positive and negative examples for training
    pos_examples, neg_examples = set_mnist_pos_neg(1, 0, train_labels, train_features)

    for lambda_reg in lambda_values:

        # Store accuracies for multiple runs
        accuracies = []
        
        # Train the model with the current set of hyperparameters
        # Repeat the experiment 'num_runs' times and calculate the average accuracy
        for run in range(num_runs):
            # Train model with current lambda_reg
            weights = train(pos_examples, neg_examples, num_iterations=500, lambda_reg=lambda_reg)
            # Evaluate accuracy on the test set
            val_acc = accuracy(weights, val_features, val_labels)
            accuracies.append(val_acc)
        
        # Calculate the average accuracy for this combination of lambda_reg and num_iterations
        avg_accuracy = np.mean(accuracies)
        results[(lambda_reg)] = avg_accuracy
        print(f"Average accuracy for lambda_reg={lambda_reg}: {avg_accuracy:.4f}")
        
        # Track the best model
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_lambda = lambda_reg
    
    # Output the best parameters
    print(f"\nBest lambda_reg: {best_lambda}, Best accuracy: {best_accuracy:.4f}")
    
    return results

# Run the experiments
# results = run_experiments_using_validation()

# Set positive and negative examples (binary classification)
pos_examples, neg_examples = set_mnist_pos_neg(1, 0, train_labels, train_features)
# Train the perceptron model
wts = train(pos_examples, neg_examples, 500, 0.15)
# Compute the accuracy on the test data
print(f"Test Accuracy: {accuracy(wts, test_features, test_labels):.4f}")