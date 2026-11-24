import numpy as np


class MultipleLinearRegressionGD:

    def __init__(self, learning_rate=0.01, epochs=10000):

        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite.")

        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Epochs must be a positive integer.")

        # Hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Model parameters
        self.weights = None
        self.bias = 0.0

        # Track training loss
        self.loss_history = []

    def fit(self, X, y):
        """
        Train Multiple Linear Regression using
        Batch Gradient Descent.
        """

        # Convert inputs into NumPy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Validate dimensions
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("Invalid input dimensions.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Sample count mismatch.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Dataset cannot be empty.")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("Dataset must contain finite values.")

        # Extract number of samples and features
        n_samples, n_features = X.shape

        # Initialize one weight for each feature
        self.weights = np.zeros(n_features)

        # Initialize bias
        self.bias = 0.0

        # Reset loss history
        self.loss_history = []

        # ======================================
        # Gradient Descent Training Loop
        # ======================================

        for epoch in range(self.epochs):

            # Step 1: Forward propagation
            y_pred = X @ self.weights + self.bias

            # Step 2: Calculate prediction errors
            error = y_pred - y

            # Step 3: Calculate weight gradients
            dw = (2 / n_samples) * (X.T @ error)

            # Step 4: Calculate bias gradient
            db = (2 / n_samples) * np.sum(error)

            # Step 5: Update weights
            self.weights -= self.learning_rate * dw

            # Step 6: Update bias
            self.bias -= self.learning_rate * db

            # Step 7: Generate updated predictions
            updated_predictions = (
                X @ self.weights + self.bias
            )

            # Step 8: Calculate updated cost
            loss = np.mean((updated_predictions - y) ** 2)

            # Detect numerical divergence
            if not np.isfinite(loss):
                raise FloatingPointError(
                    "Training diverged. Reduce the learning rate."
                )

            # Step 9: Store loss
            self.loss_history.append(loss)

            # Step 10: Display training progress
            if (epoch + 1) % 1000 == 0 or epoch == 0:

                print(
                    f"Epoch: {epoch + 1:5d} | "
                    f"Loss: {loss:.6f} | "
                    f"Weights: {np.round(self.weights, 6)} | "
                    f"Bias: {self.bias:.6f}"
                )

        return self

    def predict(self, X):
        """
        Predict using the learned weights and bias.
        """

        if self.weights is None:
            raise ValueError("Train the model first.")

        X = np.asarray(X, dtype=float)

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

model = MultipleLinearRegressionGD(
    learning_rate=0.01,
    epochs=10000
)


# ==========================================
# 3. Train model
# ==========================================

model.fit(X, y)


# ==========================================
# 4. Display learned parameters
# ==========================================

print("\nTraining completed")

print("Final weights:", np.round(model.weights, 4))
print("Final bias:", round(model.bias, 4))
print("Final loss:", round(model.loss_history[-1], 4))


# ==========================================
# 5. Generate predictions
# ==========================================

y_pred = model.predict(X)

print("Actual values:", y)
print("Predicted values:", np.round(y_pred, 4))


# ==========================================
# 6. Predict unseen data
# ==========================================

X_new = np.array([[7, 4]])

prediction = model.predict(X_new)

print("New prediction:", round(prediction[0], 4))
