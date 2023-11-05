import numpy as np  # Importing the necessary library (NumPy) for numerical operations.
import time  # Importing the time library to measure the execution time.

start = time.time()  # Record the starting time to measure execution time.


# Define the sigmoid activation function.
def f(x):
    return 1 / (1 + np.exp(-x))


# Define the derivative of the sigmoid activation function.
def df(x):
    return x * (1 - x)


# Define the input data (x_in) and the expected output (y_out).
x_in = np.array([[4], [6]])  # Input data
y_out = np.array([[0.4], [0.6]])  # Expected output

# Get the number of input and output samples.
N_in = len(x_in)
N_out = len(y_out)

# Define the number of neurons in each layer.
N1 = 3  # Number of neurons in the first hidden layer
N2 = 3  # Number of neurons in the second hidden layer

# Set the learning rate (lmb) for gradient descent.
lmb = 0.1

# Initialize weights for each layer with random values.
w1 = np.random.rand(N1, N_in)
w2 = np.random.rand(N2, N1)
w3 = np.random.rand(N_out, N2)

# Training loop for 5000 epochs.
for i in range(5000):
    rez1 = np.dot(w1, x_in)  # Calculate the weighted sum of the first hidden layer.
    A = f(rez1)  # Apply the activation function to obtain the output of the first hidden layer.

    rez2 = np.dot(w2, A)  # Calculate the weighted sum of the second hidden layer.
    B = f(rez2)  # Apply the activation function to obtain the output of the second hidden layer.

    rez3 = np.dot(w3, B)  # Calculate the weighted sum of the output layer.
    Y = f(rez3)  # Apply the activation function to obtain the final network output.

    e = Y - y_out  # Calculate the error.

    # Backpropagation to adjust the weights.
    delta3 = e * df(Y)
    w3 = w3 - lmb * np.dot(delta3, B.T)

    delta2 = np.dot(w3.T, delta3) * df(B)
    w2 = w2 - lmb * np.dot(delta2, A.T)

    delta1 = np.dot(w2.T, delta2) * df(A)
    w1 = w1 - lmb * np.dot(delta1, x_in.T)

print(Y)  # Output after training
#print(w1)  # Weights of the first layer
#print(w2)  # Weights of the second layer
#print(w3)  # Weights of the output layer

end = time.time()
print("time = ", end - start)  # Measure and print the execution time.
