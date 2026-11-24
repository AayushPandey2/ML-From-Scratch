import numpy as np


class RidgeRegressionGD:

    def __init__(
        self,
        alpha=1.0,
        learning_rate=0.05,
        epochs=1000
    ):

        # Validate hyperparameters
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("Alpha must be non-negative.")

        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Epochs must be a positive integer.")

        # Hyperparameters
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Learnable parameters
        self.weights = None
        self.bias = 0.0

        # Training history
        self.loss_history = []
        self.mse_history = []
        self.penalty_history = []


    def _calculate_loss(self, X, y):

        # Generate predictions
        y_pred = X @ self.weights + self.bias

        # Calculate Mean Squared Error
        mse = np.mean((y_pred - y) ** 2)

        # Calculate L2 penalty
        penalty = (
            self.alpha / len(y)
            * np.sum(self.weights ** 2)
        )

        # Calculate total regularized loss
        total_loss = mse + penalty

        return mse, penalty, total_loss


    def fit(self, X, y):
        """
        Train Ridge Regression using
        Batch Gradient Descent.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Validate input data
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("Invalid input dimensions.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Sample count mismatch.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Dataset cannot be empty.")

        if not np.all(np.isfinite(X)):
            raise ValueError("Invalid feature values.")

        if not np.all(np.isfinite(y)):
            raise ValueError("Invalid target values.")

        # Step 1: Extract dimensions
        n_samples, n_features = X.shape

        # Step 2: Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Reset training history
        self.loss_history = []
        self.mse_history = []
        self.penalty_history = []

        # Store initial loss
        mse, penalty, loss = self._calculate_loss(X, y)

        self.mse_history.append(mse)
        self.penalty_history.append(penalty)
        self.loss_history.append(loss)

        # ======================================
        # Gradient Descent Training Loop
        # ======================================

        for epoch in range(self.epochs):

            # Step 3: Forward pass
            y_pred = X @ self.weights + self.bias

            # Step 4: Calculate errors
            error = y_pred - y

            # Step 5: Calculate prediction gradient
            mse_gradient = (
                (2 / n_samples)
                * (X.T @ error)
            )

            # Step 6: Calculate L2 gradient
            regularization_gradient = (
                (2 * self.alpha / n_samples)
                * self.weights
            )

            # Step 7: Combine gradients
            dw = (
                mse_gradient
                + regularization_gradient
            )

            # Step 8: Calculate bias gradient
            db = (2 / n_samples) * np.sum(error)

            # Step 9: Update model parameters
            self.weights -= self.learning_rate * dw

            self.bias -= self.learning_rate * db

            # Step 10: Calculate updated loss
            mse, penalty, loss = self._calculate_loss(X, y)

            # Detect numerical divergence
            if not np.isfinite(loss):
                raise FloatingPointError(
                    "Training diverged. Reduce learning rate."
                )

            # Step 11: Store training history
            self.mse_history.append(mse)
            self.penalty_history.append(penalty)
            self.loss_history.append(loss)

            # Step 12: Display training progress
            if epoch == 0 or (epoch + 1) % 100 == 0:

                print(
                    f"Epoch: {epoch + 1:4d} | "
                    f"MSE: {mse:.6f} | "
                    f"Penalty: {penalty:.6f} | "
                    f"Loss: {loss:.6f}"
                )

        return self


    def predict(self, X):

        if self.weights is None:
            raise ValueError("Train the model first.")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != len(self.weights):
            raise ValueError("Incorrect number of features.")

        if not np.all(np.isfinite(X)):
            raise ValueError("Invalid feature values.")

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

model = RidgeRegressionGD(
    alpha=2.0,
    learning_rate=0.05,
    epochs=1000
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

print("Final MSE:", round(model.mse_history[-1], 4))

print(
    "Final penalty:",
    round(model.penalty_history[-1], 4)
)

print("Final loss:", round(model.loss_history[-1], 4))


# ==========================================
# 5. Generate predictions
# ==========================================

y_pred = model.predict(X)

print("Actual values:", y)

print("Predicted values:", np.round(y_pred, 4))


# ==========================================
# 6. Predict unseen sample
# ==========================================

prediction = model.predict([[3, 2]])

print("New prediction:", round(prediction[0], 4))
