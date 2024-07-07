# Neural Network and Deep Learning Course Tasks

## Task 1: Perceptron and Adaline Implementation

### Overview
This task involves the implementation of two classic machine learning models: Perceptron and Adaline. These models are implemented from scratch using Python and NumPy.

### Perceptron
The perceptron is a simple linear classifier that learns a set of weights for input features to classify data points into binary categories. It updates weights iteratively to minimize classification errors.

#### Functions Implemented:
- **signum(x)**: Activation function to determine the output class based on the sign of the input.
- **perceptron_lr(W, x1, x2, lr, epochs, target, b, flag)**: Function to train a perceptron using a learning rate and specified number of epochs.
- **adaline(W, x1, x2, lr, epochs, target, MSEthreshold, b, flag)**: Function to train an Adaline model using a learning rate and stopping criteria based on Mean Squared Error (MSE).

### Usage
To use these implementations, provide input data `x1`, `x2`, target labels `target`, and specify parameters like learning rate (`lr`) and number of epochs (`epochs`). Adjust the `flag` parameter as needed for bias handling.

---

## Task 2: Neural Network with Backpropagation

### Overview
This task involves the implementation of a neural network using backpropagation for training. The neural network is designed to have customizable architecture with multiple hidden layers and activation functions.

### Neural Network Architecture
The neural network architecture includes:
- Customizable number of hidden layers and neurons per layer.
- Activation functions include sigmoid and tanh functions.

#### Functions Implemented:
- **init_weight(HiddenLayers, NeuronsPerLayer, flag)**: Function to initialize random weights for input, hidden, and output layers.
- **sigmoidAF(x)**: Sigmoid activation function.
- **TanhAF(x)**: Hyperbolic tangent activation function.
- **forward_step(input_w, hidden_w, output_w, X_train, HiddenLayers, NeuronsPerLayer, flag, actFn)**: Forward propagation through the network to compute outputs.
- **Back_propagation(X_train, ytrain, HiddenLayers, NeuronsPerLayer, lr, epochs_Number, flag, actFn)**: Backpropagation algorithm to update weights based on errors.
- **test(X_test, ytest, input_w, hidden_w, output_w, HiddenLayers, NeuronsPerLayer, flag, actFn)**: Function to test the neural network model on test data and calculate confusion matrix.

### Usage
To utilize the neural network, specify parameters such as the number of hidden layers (`HiddenLayers`), neurons per layer (`NeuronsPerLayer`), learning rate (`lr`), and activation function (`actFn`). Ensure input data (`X_train`, `ytrain`, `X_test`, `ytest`) are appropriately formatted.

 
