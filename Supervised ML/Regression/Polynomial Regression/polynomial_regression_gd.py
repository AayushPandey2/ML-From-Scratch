import numpy as np


class PolynomialRegressionGD:

    def __init__(
        self,
        degree=2,
        learning_rate=0.01,
        epochs=5000
    ):

        # Validate hyperparameters
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("Degree must be a positive integer.")

        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Invalid learning rate.")

        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Epochs must be a positive integer.")

        # Hyperparameters
        self.degree = degree
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Learnable parameters
        self.weights = None
        self.bias = 0.0

        # Track loss during training
        self.loss_history = []


    def _transform(self, X):
        """
        Generate polynomial features.
        """

        X = np.asarray(X, dtype=float)
        X = np.atleast_1d(X)

        if X.ndim != 1:
            raise ValueError("X must be a 1D array.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain finite values.")

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
        Batch Gradient Descent.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Validate inputs
        if X.ndim != 1 or y.ndim != 1:
            raise ValueError("X and y must be 1D arrays.")

        if len(X) != len(y) or len(X) == 0:
            raise ValueError("Invalid dataset size.")

        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain finite values.")

        # Step 1: Transform original features
        X_poly = self._transform(X)

        # Step 2: Extract matrix dimensions
        n_samples, n_features = X_poly.shape

        # Step 3: Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Reset loss history
        self.loss_history = []

        # Record initial loss
        initial_predictions = (
            X_poly @ self.weights + self.bias
        )

        initial_loss = np.mean(
            (initial_predictions - y) ** 2
        )

        self.loss_history.append(initial_loss)

        # ======================================
        # Gradient Descent Training Loop
        # ======================================

        for epoch in range(self.epochs):

            # Step 4: Forward pass
            y_pred = X_poly @ self.weights + self.bias

            # Step 5: Calculate errors
            error = y_pred - y

            # Step 6: Calculate weight gradients
            dw = (
                (2 / n_samples)
                * (X_poly.T @ error)
            )

            # Step 7: Calculate bias gradient
            db = (2 / n_samples) * np.sum(error)

            # Step 8: Update weights
            self.weights -= self.learning_rate * dw

            # Step 9: Update bias
            self.bias -= self.learning_rate * db

            # Step 10: Calculate updated predictions
            updated_predictions = (
                X_poly @ self.weights + self.bias
            )

            # Step 11: Calculate updated loss
            loss = np.mean(
                (updated_predictions - y) ** 2
            )

            # Detect numerical divergence
            if not np.isfinite(loss):
                raise FloatingPointError(
                    "Training diverged. Reduce learning rate "
                    "or scale polynomial features."
                )

            # Step 12: Store loss
            self.loss_history.append(loss)

            # Step 13: Display training progress
            if (epoch + 1) % 500 == 0 or epoch == 0:

                print(
                    f"Epoch: {epoch + 1:5d} | "
                    f"Loss: {loss:.6f} | "
                    f"Weights: {np.round(self.weights, 4)} | "
                    f"Bias: {self.bias:.6f}"
                )

        return self


    def predict(self, X):

        if self.weights is None:
            raise ValueError("Train the model first.")

        # Transform features
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

model = PolynomialRegressionGD(
    degree=2,
    learning_rate=0.01,
    epochs=5000
)


# ==========================================
# 3. Train model
# ==========================================

model.fit(X, y)


# ==========================================
# 4. Display final parameters
# ==========================================

print("\nTraining completed")

print("Weights:", np.round(model.weights, 4))
print("Bias:", round(model.bias, 4))
print("Initial loss:", round(model.loss_history[0], 4))
print("Final loss:", round(model.loss_history[-1], 4))


# ==========================================
# 5. Generate predictions
# ==========================================

y_pred = model.predict(X)

print("Actual values:", y)
print("Predicted values:", np.round(y_pred, 4))


# ==========================================
# 6. Predict unseen input
# ==========================================

prediction = model.predict([4])

print("Prediction for x = 4:", round(prediction[0], 4))
