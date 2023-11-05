import numpy as np  # Import the NumPy library for mathematical operations.

# Define input temperature data in Celsius (C) and their corresponding target values in Fahrenheit (F).
C = np.array([-40, -10, 0, 8, 15, 22, 38])
B = [1] * len(C)  # Create an array of ones to represent the bias term.
B = np.array(B)
F = np.array([-40, 14, 32, 46, 59, 72, 100])

lmb = 0.001  # Set the learning rate for gradient descent.

# Define the activation function and its derivative.
def f(x):
    return x

def df(x):
    return 1

# Initialize the weight vector.
w = np.array([0.5, 0.5])

# Training the model using stochastic gradient descent.
for i in range(50000):
    k = np.random.randint(0, len(C))  # Randomly select a data point.
    X = np.array([C[k], B[k]])
    V = np.dot(w, X)
    y = f(V)
    e = y - F[k]  # Compute the error.
    w = w - lmb * e * df(y) * X  # Update the weight vector.

# Display the learned weights.
print(w)

# Make a prediction for a new input temperature in Celsius (100°C).
X = np.array([100, 1])
V = np.dot(w, X)
y = f(V)

# Display the predicted temperature in Fahrenheit.
print(y)
