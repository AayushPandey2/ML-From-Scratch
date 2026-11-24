        """
        Learn the weight and bias using Ordinary Least Squares.

        Parameters:
            X: Independent variable (input feature)
            y: Dependent variable (target)
        """
import numpy as np
class SimpleLinearRegression:
    def __init__(self):
        self.weight = None
        self.bias = None
    def fit(self, X, y):
       

        # Convert inputs into NumPy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Validate input data
        if X.ndim != 1 or y.ndim != 1:
            raise ValueError("X and y must be 1D arrays.")

        if len(X) != len(y) or len(X) < 2:
            raise ValueError("Invalid dataset size.")

        # Step 1: Calculate the means
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        # Step 2: Calculate deviations from the means
        X_deviation = X - X_mean
        y_deviation = y - y_mean

        # Step 3: Calculate the numerator
        numerator = np.sum(X_deviation * y_deviation)

        # Step 4: Calculate the denominator
        denominator = np.sum(X_deviation ** 2)

        # A constant feature cannot determine a unique slope
        if denominator == 0:
            raise ValueError("X must contain varying values.")

        # Step 5: Calculate weight
        self.weight = numerator / denominator

        # Step 6: Calculate bias
        self.bias = y_mean - self.weight * X_mean

        return self

    def predict(self, X):
        """Predict target values using y = wx + b."""

        if self.weight is None:
            raise ValueError("Train the model before prediction.")

        X = np.asarray(X, dtype=float)

        return self.weight * X + self.bias


# ==========================================
# Dataset
# ==========================================

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)


# ==========================================
# Model training
# ==========================================

model = SimpleLinearRegression()
model.fit(X, y)


# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(X)

print("Weight:", round(model.weight, 4))
print("Bias:", round(model.bias, 4))
print("Actual values:", y)
print("Predicted values:", np.round(y_pred, 4))

# Predict an unseen input
new_prediction = model.predict(6)

print("Prediction for X = 6:", round(new_prediction, 4))
