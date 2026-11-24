import numpy as np


class MultipleLinearRegression:

    def __init__(self):

        # Model parameters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the model using Ordinary Least Squares.
        """

        # Convert inputs into NumPy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Validate input dimensions
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have equal sample counts.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Dataset cannot be empty.")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("Dataset must contain finite values.")

        # Number of training samples
        n_samples = X.shape[0]

        # Create a column of ones for the bias
        ones = np.ones((n_samples, 1))

        # Combine bias column with feature matrix
        X_b = np.hstack((ones, X))

        # Calculate optimal parameters
        # theta = pseudoinverse(X_b) @ y
        theta = np.linalg.pinv(X_b) @ y

        # Extract bias
        self.bias = theta[0]

        # Extract feature weights
        self.weights = theta[1:]

        return self

    def predict(self, X):
        """
        Predict target values using X @ weights + bias.
        """

        if self.weights is None:
            raise ValueError("Train the model first.")

        X = np.asarray(X, dtype=float)

        # Allow prediction for a single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise ValueError("Incorrect number of features.")

        return X @ self.weights + self.bias


# ==========================================
# 1. Dataset
# ==========================================

X = np.array([
    [1, 2],
    [2, 1],
    [3, 3],
    [4, 1],
    [5, 3],
    [6, 2]
], dtype=float)

y = np.array([14, 11, 23, 18, 28, 29], dtype=float)


# ==========================================
# 2. Initialize model
# ==========================================

model = MultipleLinearRegression()


# ==========================================
# 3. Train model
# ==========================================

model.fit(X, y)


# ==========================================
# 4. Display learned parameters
# ==========================================

print("Weights:", np.round(model.weights, 4))
print("Bias:", round(model.bias, 4))


# ==========================================
# 5. Generate predictions
# ==========================================

y_pred = model.predict(X)

print("Actual values:", y)
print("Predicted values:", np.round(y_pred, 4))


# ==========================================
# 6. Predict an unseen sample
# ==========================================

X_new = np.array([[7, 4]])

prediction = model.predict(X_new)

print("New prediction:", round(prediction[0], 4))
