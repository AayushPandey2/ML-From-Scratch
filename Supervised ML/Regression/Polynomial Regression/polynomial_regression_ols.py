import numpy as np


class PolynomialRegression:

    def __init__(self, degree=2):

        # Validate polynomial degree
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("Degree must be a positive integer.")

        self.degree = degree

        # Learnable model parameters
        self.weights = None
        self.bias = None


    def _transform(self, X):
        """
        Convert original inputs into polynomial features.

        Example:
        x = 2, degree = 3

        [2, 4, 8]
        """

        X = np.asarray(X, dtype=float)

        # Allow scalar input during prediction
        X = np.atleast_1d(X)

        if X.ndim != 1:
            raise ValueError("X must be a 1D array.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain finite values.")

        # Generate polynomial features
        X_poly = np.column_stack([
            X ** i
            for i in range(1, self.degree + 1)
        ])

        if not np.all(np.isfinite(X_poly)):
            raise ValueError("Polynomial features overflowed.")

        return X_poly


    def fit(self, X, y):
        """
        Train Polynomial Regression using
        Ordinary Least Squares.
        """

        # Convert inputs
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Validate training data
        if X.ndim != 1 or y.ndim != 1:
            raise ValueError("X and y must be 1D arrays.")

        if len(X) != len(y):
            raise ValueError("Sample count mismatch.")

        if len(X) == 0:
            raise ValueError("Dataset cannot be empty.")

        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain finite values.")

        # Step 1: Generate polynomial features
        X_poly = self._transform(X)

        # Step 2: Determine sample count
        n_samples = X_poly.shape[0]

        # Step 3: Create bias column
        ones = np.ones((n_samples, 1))

        # Step 4: Construct augmented matrix
        X_b = np.hstack((ones, X_poly))

        # Step 5: Calculate optimal parameters
        theta = np.linalg.pinv(X_b) @ y

        # Step 6: Extract bias
        self.bias = theta[0]

        # Step 7: Extract polynomial weights
        self.weights = theta[1:]

        return self


    def predict(self, X):
        """
        Predict target values using learned
        polynomial coefficients.
        """

        if self.weights is None:
            raise ValueError("Train the model first.")

        # Apply the same feature transformation
        X_poly = self._transform(X)

        # Generate predictions
        return X_poly @ self.weights + self.bias


# ==========================================
# 1. Create dataset
# ==========================================

X = np.array(
    [-3, -2, -1, 0, 1, 2, 3],
    dtype=float
)

y = np.array(
    [-2, -3, -1, 4, 5, 9, 16],
    dtype=float
)


# ==========================================
# 2. Initialize model
# ==========================================

model = PolynomialRegression(degree=2)


# ==========================================
# 3. Train model
# ==========================================

model.fit(X, y)


# ==========================================
# 4. Display learned parameters
# ==========================================

print("Polynomial degree:", model.degree)
print("Weights:", np.round(model.weights, 4))
print("Bias:", round(model.bias, 4))


# ==========================================
# 5. Generate predictions
# ==========================================

y_pred = model.predict(X)

print("\nActual values:", y)
print("Predicted values:", np.round(y_pred, 4))


# ==========================================
# 6. Calculate training loss
# ==========================================

mse = np.mean((y_pred - y) ** 2)

print("MSE:", round(mse, 4))


# ==========================================
# 7. Predict an unseen input
# ==========================================

X_new = np.array([4])

prediction = model.predict(X_new)

print("Prediction for x = 4:", round(prediction[0], 4))
