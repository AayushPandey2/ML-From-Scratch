import numpy as np


def soft_threshold(z, threshold):

    return np.sign(z) * np.maximum(
        np.abs(z) - threshold,
        0.0
    )


class LassoRegressionPGD:

    def __init__(
        self,
        alpha=1.0,
        learning_rate=None,
        epochs=10000,
        tolerance=1e-9
    ):

        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("Invalid alpha.")

        if learning_rate is not None:
            if (
                not np.isfinite(learning_rate)
                or learning_rate <= 0
            ):
                raise ValueError("Invalid learning rate.")

        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Invalid epoch count.")

        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Invalid tolerance.")

        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance

        self.weights = None
        self.bias = None
        self.learning_rate_ = None

        self.loss_history = []
        self.converged = False
        self.n_iterations = 0


    def fit(self, X, y):

        # Convert inputs
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Validate inputs
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

        # Step 2: Center the dataset
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)

        X_centered = X - X_mean
        y_centered = y - y_mean

        # Step 3: Initialize weights
        weights = np.zeros(n_features)

        # Step 4: Calculate a stable learning rate
        L = (
            2 / n_samples
            * np.linalg.norm(X_centered, ord=2) ** 2
        )

        safe_rate = 1 / L if L > 0 else 1.0

        if self.learning_rate is None:

            eta = safe_rate

        else:

            eta = self.learning_rate

            if L > 0 and eta > safe_rate:
                raise ValueError(
                    "Learning rate exceeds the "
                    "selected stability bound."
                )

        self.learning_rate_ = eta

        # Step 5: Calculate soft-thresholding strength
        threshold = eta * self.alpha / n_samples

        # Reset training statistics
        self.loss_history = [
            np.mean(y_centered ** 2)
        ]

        self.converged = False
        self.n_iterations = 0

        # ======================================
        # Proximal Gradient Descent Loop
        # ======================================

        for epoch in range(self.epochs):

            # Step 6: Forward pass
            y_pred = X_centered @ weights

            # Step 7: Calculate prediction errors
            error = y_pred - y_centered

            # Step 8: Calculate MSE gradient
            dw = (
                (2 / n_samples)
                * (X_centered.T @ error)
            )

            # Step 9: Ordinary gradient update
            z = weights - eta * dw

            # Step 10: Apply L1 soft-thresholding
            weights = soft_threshold(
                z,
                threshold
            )

            # Step 11: Calculate updated predictions
            y_pred = X_centered @ weights

            # Step 12: Calculate MSE
            mse = np.mean(
                (y_pred - y_centered) ** 2
            )

            # Step 13: Calculate L1 penalty
            penalty = (
                self.alpha / n_samples
                * np.sum(np.abs(weights))
            )

            # Step 14: Calculate total objective
            loss = mse + penalty

            if not np.isfinite(loss):
                raise FloatingPointError(
                    "Training diverged."
                )

            self.loss_history.append(loss)
            self.n_iterations = epoch + 1

            # Step 15: Check optimality conditions
            gradient = (
                (2 / n_samples)
                * X_centered.T
                @ (y_pred - y_centered)
            )

            l1_strength = self.alpha / n_samples

            violation = np.where(
                weights != 0,

                np.abs(
                    gradient
                    + l1_strength * np.sign(weights)
                ),

                np.maximum(
                    np.abs(gradient) - l1_strength,
                    0.0
                )
            )

            if np.max(violation) <= self.tolerance:

                self.converged = True
                break

        # Store learned weights
        self.weights = weights

        # Recover unregularized bias
        self.bias = (
            y_mean - X_mean @ weights
        )

        return self


    def predict(self, X):

        if self.weights is None:
            raise ValueError("Train the model first.")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError("Invalid input dimensions.")

        if X.shape[1] != len(self.weights):
            raise ValueError("Incorrect feature count.")

        if not np.all(np.isfinite(X)):
            raise ValueError("Invalid feature values.")

        return X @ self.weights + self.bias


# ==========================================
# Example usage
# ==========================================

if __name__ == "__main__":

    X = np.array([
        [-2, -1],
        [-1,  2],
        [ 1, -2],
        [ 2,  1]
    ], dtype=float)

    y = np.array(
        [-4.5, 0, 4, 8.5],
        dtype=float
    )

    # Initialize model
    model = LassoRegressionPGD(
        alpha=10.0,
        learning_rate=0.1,
        epochs=10000
    )

    # Train model
    model.fit(X, y)

    # Generate predictions
    y_pred = model.predict(X)

    # Display results
    print("Weights:", np.round(model.weights, 4))
    print("Bias:", round(model.bias, 4))

    print("Actual values:", y)
    print("Predictions:", np.round(y_pred, 4))

    print("Initial loss:", model.loss_history[0])
    print("Final loss:", round(model.loss_history[-1], 4))

    print("MSE:", round(
        np.mean((y_pred - y) ** 2),
        4
    ))

    print("Converged:", model.converged)

    print(
        "New prediction:",
        round(model.predict([[3, 2]])[0], 4)
    )
