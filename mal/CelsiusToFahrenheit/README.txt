pip install numpy

1. Import necessary libraries, including NumPy.
2. Define input data: Celsius temperatures (C) and their corresponding Fahrenheit values (F).
3. Initialize a bias term array (B) for linear regression.
4. Set the learning rate (lmb) for gradient descent.
5. Define the activation function (f(x)) and its derivative (df(x)).
6. Initialize the weight vector (w) for the linear regression model.
7. Train the model using stochastic gradient descent (SGD) for a specified number of iterations (10,000 in this case).
   a. Randomly select a data point (k).
   b. Compute the predicted output (y) based on the current weights and input data (X).
   c. Calculate the error (e) between the predicted and actual target values.
   d. Update the weight vector (w) using the gradient descent rule.
8. Display the learned weights (w) after training.
9. Make a prediction for a new input temperature (100°C).
10. Display the predicted temperature in Fahrenheit (y).

This program is essentially performing linear regression to learn the relationship between Celsius and Fahrenheit temperatures. The model is trained using stochastic gradient descent to find the optimal weights.