import numpy as np


class RidgeRegression:

    def __init__(self, alpha=1.0):

        # Validate regularization strength
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("Alpha must be non-negative and finite.")

        # Regularization hyperparameter
        self.alpha = alpha

        # Learnable model parameters
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        """
        Train Ridge Regression using the closed-form
        mathematical solution.
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
            raise ValueError("Sample count mismatch.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Dataset cannot be empty.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains invalid values.")

        if not np.all(np.isfinite(y)):
            raise ValueError("y contains invalid values.")

        # Step 1: Extract dimensions
        n_samples, n_features = X.shape

        # Step 2: Calculate feature and target means
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)

        # Step 3: Center the dataset
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Step 4: Create identity matrix
        identity_matrix = np.eye(n_features)

        # Step 5: Construct regularized matrix
        A = (
            X_centered.T @ X_centered
            + self.alpha * identity_matrix
        )

        # Step 6: Construct right-hand side
        B = X_centered.T @ y_centered

        # Step 7: Calculate optimal weights
        if self.alpha == 0:

            # Ordinary Least Squares fallback
            self.weights = np.linalg.lstsq(
                X_centered,
                y_centered,
                rcond=None
            )[0]

        else:

            # Solve the regularized linear system
            self.weights = np.linalg.solve(A, B)

        # Step 8: Calculate unregularized bias
        self.bias = (
            y_mean - X_mean @ self.weights
        )

        return self


    def predict(self, X):
        """
        Generate predictions using learned parameters.
        """

        if self.weights is None:
            raise ValueError("Train the model first.")

        X = np.asarray(X, dtype=float)

        # Allow a single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != len(self.weights):
            raise ValueError("Incorrect number of features.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains invalid values.")

        return X @ self.weights + self.bias


# ==========================================
# 1. Dataset
# ==========================================

X = np.array([
    [-2, -1],
    [-1,  2],
    [ 1, -2],
    [ 2,  1]
], dtype=float)

y = np.array([-8, 7, -3, 12], dtype=float)


# ==========================================
# 2. Initialize model
# ==========================================

model = RidgeRegression(alpha=2.0)


# ==========================================
# 3. Train model
# ==========================================

model.fit(X, y)


# ==========================================
# 4. Display learned parameters
# ==========================================

print("Alpha:", model.alpha)
print("Weights:", np.round(model.weights, 4))
print("Bias:", round(model.bias, 4))


# ==========================================
# 5. Generate predictions
# ==========================================

y_pred = model.predict(X)

print("Actual values:", y)
print("Predicted values:", np.round(y_pred, 4))


# ==========================================
# 6. Calculate training error
# ==========================================

mse = np.mean((y_pred - y) ** 2)

print("MSE:", round(mse, 4))


# ==========================================
# 7. Calculate regularization penalty
# ==========================================

penalty = (
    model.alpha / len(y)
    * np.sum(model.weights ** 2)
)

print("Regularization penalty:", round(penalty, 4))

print("Total loss:", round(mse + penalty, 4))


# ==========================================
# 8. Predict an unseen sample
# ==========================================

X_new = np.array([[3, 2]])

prediction = model.predict(X_new)

print("New prediction:", round(prediction[0], 4))
