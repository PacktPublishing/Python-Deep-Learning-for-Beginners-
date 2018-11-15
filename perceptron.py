import numpy as np

# Linear regression
prediction = input_1 * weight_1 + input_2 * weight_2
prediction = np.dot([input_1, input_2], [weight_1, weight_2])
prediction = np.dot(inputs, weights)
prediction = LinearRegression().fit(inputs, outputs).predict(inputs)

# Perceptron
def predict_perceptron(inputs, weights):
    if np.dot(inputs, weights) > 0:
        return 1
    else:
        return 0


def fit_perceptron(inputs, output, epochs, learning_rate):
    weights = np.zeros(inputs.shape[0])  # initialize the weight vector
    for t in range(epochs):
        for i, x in enumerate(inputs):  # iterate over each example
            # check if example is misclassified
            if (np.dot(inputs[i], weights) * output[i]) <= 0:
                # misclassified! let's apply our learning rule
                weights = weights + learning_rate * inputs[i] * output[i]
    return weights


weights = perceptron_sgd(X, y, 100, 1)
print(weights)
