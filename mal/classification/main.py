import numpy as np  # Import the NumPy library for numerical operations
import matplotlib.pyplot as plt  # Import the Matplotlib library for plotting

# Define an activation function that returns 1 if input is greater than 0, else 0
def act_func(x):
    return 0 if x <= 0 else 1

# Define a forward_hidden function to compute the output of hidden layer nodes
def forward_hidden(w, x):
    y = []
    for w_el in w:
        y.append(forward(w_el, x))
    return y

# Define a forward function to compute the output of a single layer
def forward(w, x):
    summ = np.dot(w, x)
    y = [act_func(x) for x in summ]
    return y

n = 200  # Number of data points

# Generate random data points for x1 and x2
x1 = np.random.random(n)
x2 = np.random.random(n)

# Create a constant input (bias) of 1 for all data points
x3 = [1] * n

# Combine x1, x2, and x3 into an input array x
x = np.array([x1, x2, x3])

# Define weight vectors for the hidden layer
w1 = np.array([1, 1, -1.5])
w2 = np.array([1, 1, -0.5])
w3 = np.array([-1, 1, -0.5])
w4 = np.array([-1, 1, 0.5])

# Combine the weight vectors into a weight matrix for the hidden layer
w_hidden = np.array([w1, w2, w3, w4])

# Define a weight vector for the output layer
w_result = np.array([-1, 1, -1, 1, -1])

# Compute the results of the hidden layer
result_hidden = forward_hidden(w_hidden, x)
result_hidden.append(x3)  # Append the constant input

# Compute the final result using the output layer
result = forward(w_result, result_hidden)

# Plot the results
for i in range(n):
    if result[i] > 0:
        plt.scatter(x1[i], x2[i], c="green")  # Plot data points as green for positive results
    else:
        plt.scatter(x1[i], x2[i], c="red")  # Plot data points as red for negative results

# Plot decision boundaries
plt.plot([0.5, 1], [1, 0.5])
plt.plot([0, 0.5], [0.5, 0])
plt.plot([0, 0.5], [0.5, 1])
plt.plot([0.5, 1], [0, 0.5])

# Display the plot
plt.show()
