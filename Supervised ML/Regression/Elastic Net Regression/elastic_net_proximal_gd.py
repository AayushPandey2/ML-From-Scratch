import numpy as np


def soft_threshold(z, threshold):

    return np.sign(z) * np.maximum(
        np.abs(z) - threshold,
        0.0
    )


class ElasticNetRegressionGD:

    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.5,
        learning_rate=None,
        epochs=10000,
        tolerance=1e-9
    ):

        # Validate hyperparameters
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("Invalid alpha.")

        if not 0 <= l1_ratio <= 1:
            raise ValueError("Invalid L1 ratio.")

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

        # Hyperparameters
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance

        # Regularization strengths
        self.lambda1 = alpha * l1_ratio
        self.lambda2 = alpha * (1 - l1_ratio)

        # Model parameters
        self.weights = None
        self.bias = None

        self.learning_rate_ = None

        # Training statistics
        self.loss_history = []
        self.converged = False
        self.n_iterations = 0


    def fit(self, X, y):

        # Convert inputs into NumPy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Validate dataset
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

        # Step 2: Calculate means
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)

        # Step 3: Center dataset
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Step 4: Initialize parameters
        weights = np.zeros(n_features)

        # Step 5: Calculate Lipschitz constant
        L = (
            2 / n_samples
            * (
                np.linalg.norm(
                    X_centered,
                    ord=2
                ) ** 2
                + self.lambda2
            )
        )

        # Select stable learning rate
        safe_rate = 1 / L if L > 0 else 1.0

        if self.learning_rate is None:

            eta = safe_rate

        else:

            eta = self.learning_rate

            if L > 0 and eta > safe_rate:

                raise ValueError(
                    "Learning rate exceeds stability bound."
                )

        self.learning_rate_ = eta

        # Step 6: Calculate L1 threshold
        threshold = (
            eta * self.lambda1 / n_samples
        )

        # Reset training history
        self.loss_history = [
            np.mean(y_centered ** 2)
        ]

        self.converged = False
        self.n_iterations = 0

        # ======================================
        # Proximal Gradient Descent Loop
        # ======================================

        for epoch in range(self.epochs):

            # Step 7: Forward pass
            y_pred = X_centered @ weights

            # Step 8: Calculate prediction errors
            error = y_pred - y_centered

            # Step 9: Calculate MSE gradient
            mse_gradient = (
                2 / n_samples
                * (X_centered.T @ error)
            )

            # Step 10: Calculate L2 gradient
            l2_gradient = (
                2 * self.lambda2 / n_samples
                * weights
            )

            # Step 11: Combine gradients
            dw = mse_gradient + l2_gradient

            # Step 12: Gradient Descent update
            z = weights - eta * dw

            # Step 13: Apply L1 soft-thresholding
            weights = soft_threshold(
                z,
                threshold
            )

            # Step 14: Calculate updated predictions
            y_pred = X_centered @ weights

            # Step 15: Calculate MSE
            mse = np.mean(
                (y_pred - y_centered) ** 2
            )

            # Step 16: Calculate L1 penalty
            l1_penalty = (
                self.lambda1 / n_samples
                * np.sum(np.abs(weights))
            )

            # Step 17: Calculate L2 penalty
            l2_penalty = (
                self.lambda2 / n_samples
                * np.sum(weights ** 2)
            )

            # Step 18: Total regularized loss
            loss = mse + l1_penalty + l2_penalty

            if not np.isfinite(loss):
                raise FloatingPointError(
                    "Training diverged."
                )

            self.loss_history.append(loss)
            self.n_iterations = epoch + 1

            # Step 19: Check optimality conditions
            gradient = (
                2 / n_samples
                * (
                    X_centered.T
                    @ (y_pred - y_centered)
                    + self.lambda2 * weights
                )
            )

            l1_strength = self.lambda1 / n_samples

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

        # Recover intercept
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
    model = ElasticNetRegressionGD(
        alpha=12,
        l1_ratio=5 / 6,
        learning_rate=0.1,
        epochs=10000
    )

    # Train model
    model.fit(X, y)

    # Generate predictions
    y_pred = model.predict(X)

    # Display model parameters
    print("Weights:", np.round(model.weights, 4))
    print("Bias:", round(model.bias, 4))

    print("Predictions:", np.round(y_pred, 4))

    print("Initial loss:", model.loss_history[0])

    print(
        "Final loss:",
        round(model.loss_history[-1], 4)
    )

    print("Converged:", model.converged)

    print(
        "New prediction:",
        round(model.predict([[3, 2]])[0], 4)
    )
