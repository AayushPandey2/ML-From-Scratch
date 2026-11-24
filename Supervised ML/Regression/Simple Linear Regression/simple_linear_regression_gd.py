       """
        Learn the weight and bias using Gradient Descent.

        Parameters:
            X: Independent variable (input feature)
            y: Dependent variable (target)
        """
import numpy as np


class SimpleLinearRegressionGD:

    def __init__(self, learning_rate=0.05, epochs=5000):

        # Hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Model parameters
        self.weight = 0.0
        self.bias = 0.0

        # Store loss after every parameter update
        self.loss_history = []


    def fit(self, X, y):
        """Train the model using Batch Gradient Descent."""

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Validate input data
        if X.ndim != 1 or y.ndim != 1:
            raise ValueError("X and y must be 1D arrays.")

        if len(X) != len(y) or len(X) < 2:
            raise ValueError("Invalid dataset size.")

        if np.var(X) == 0:
            raise ValueError("X must contain varying values.")

        # Number of training samples
        n = len(X)

        # Initialize parameters
        self.weight = 0.0
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):

            # Step 1: Forward propagation
            y_pred = self.weight * X + self.bias

            # Step 2: Calculate prediction errors
            error = y_pred - y

            # Step 3: Calculate gradients
            dw = (2 / n) * np.sum(X * error)
            db = (2 / n) * np.sum(error)

            # Step 4: Update model parameters
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Step 5: Calculate updated loss
            updated_predictions = self.weight * X + self.bias

            loss = np.mean((updated_predictions - y) ** 2)

            self.loss_history.append(loss)

            # Display training progress
            if epoch % 500 == 0:
                print(
                    f"Epoch: {epoch + 1:5d} | "
                    f"Loss: {loss:.6f} | "
                    f"Weight: {self.weight:.6f} | "
                    f"Bias: {self.bias:.6f}"
                )

        return self


    def predict(self, X):
        """Generate predictions."""

        X = np.asarray(X, dtype=float)

        return self.weight * X + self.bias


# ==========================================
# Dataset
# ==========================================

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)


# ==========================================
# Model initialization and training
# ==========================================

model = SimpleLinearRegressionGD(
    learning_rate=0.05,
    epochs=5000
)

model.fit(X, y)


# ==========================================
# Results
# ==========================================

y_pred = model.predict(X)

print("\nTraining completed")
print("Final weight:", round(model.weight, 4))
print("Final bias:", round(model.bias, 4))
print("Final loss:", round(model.loss_history[-1], 4))
print("Predictions:", np.round(y_pred, 4))
print("Prediction for X = 6:", round(model.predict(6), 4))
