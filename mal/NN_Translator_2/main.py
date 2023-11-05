import numpy as np  # Import NumPy for numerical operations
import time  # Import the time module to measure execution time

start = time.time()  # Record the start time

# Define the input data and the desired output
x_in = np.array([[2], [2], [2]])
goal = np.array([[0.3], [0.4], [0.3]])

# Define the dimensions of the input and output
N_in = len(x_in)
N_out = len(goal)

# Define the number of neurons in the hidden layers
N1 = 4
N2 = 4

# Initialize random weights for the layers
w1 = np.random.rand(N1, N_in)
w2 = np.random.rand(N2, N1)
w3 = np.random.rand(N_out, N2)

# Set the learning rate
lmb = 0.1

# Define the sigmoid activation function and its derivative
def f(x):
    return 1 / (1 + np.exp(-x))

def df(x):
    return x * (1 - x)

# Training loop (5000 iterations)
for i in range(5000):
    w1x_in = np.dot(w1, x_in)
    y1 = f(w1x_in)
    w2y1 = np.dot(w2, y1)
    y2 = f(w2y1)
    w3y2 = np.dot(w3, y2)
    out = f(w3y2)
    e = out - goal
    delta3 = e * df(out)
    w3 = w3 - lmb * np.dot(delta3, y2.T)
    delta2 = np.dot(w3.T, delta3) * df(y2)
    w2 = w2 - lmb * np.dot(delta2, y1.T)
    delta1 = np.dot(w2.T, delta2) * df(y1)
    w1 = w1 - lmb * np.dot(delta1, x_in.T)

print(out)  # Display the final output

end = time.time()  # Record the end time
print("time = ", end - start)  # Display the execution time
