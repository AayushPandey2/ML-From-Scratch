import numpy as np


def soft_threshold(z, threshold):
    """
    Shrink values toward zero.
    """

    return np.sign(z) * np.maximum(
        np.abs(z) - threshold,
        0.0
    )


class LassoRegression:

    def __init__(
        self,
        alpha=1.0,
        max_iterations=10000,
        tolerance=1e-9
    ):

        # Validate hyperparameters
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("Alpha must be non-negative.")

        if (
            not isinstance(max_iterations, int)
            or max_iterations <= 0
        ):
            raise ValueError("Invalid iteration count.")

        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Tolerance must be positive.")

        # Hyperparameters
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Learnable parameters
        self.weights = None
        self.bias = None

        # Training statistics
        self.loss_history = []
        self.n_iterations = 0
        self.converged = False


    def fit(self, X, y):
        """
        Train Lasso Regression using Coordinate Descent.
        """

        # Convert inputs into NumPy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Validate input dimensions
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

        # Step 1: Extract dataset dimensions
        n_samples, n_features = X.shape

        # Step 2: Calculate feature and target means
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)

        # Step 3: Center the dataset
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Step 4: Initialize weights
        weights = np.zeros(n_features)

        # Step 5: Calculate squared feature norms
        feature_norms = np.sum(
            X_centered ** 2,
            axis=0
        )

        # Reset training statistics
        self.loss_history = [
            np.mean(y_centered ** 2)
        ]

        self.converged = False
        self.n_iterations = 0

        # Initialize residual
        residual = y_centered.copy()

        # ======================================
        # Coordinate Descent Training Loop
        # ======================================

        for iteration in range(self.max_iterations):

            # Optimize one feature at a time
            for j in range(n_features):

                if feature_norms[j] == 0:
                    continue

                # Extract the current feature
                feature = X_centered[:, j]

                # Remove current weight's contribution
                partial_residual = (
                    residual
                    + feature * weights[j]
                )

                # Calculate rho
                rho = feature @ partial_residual

                # Apply soft-thresholding
                new_weight = (
                    soft_threshold(
                        rho,
                        self.alpha / 2
                    )
                    / feature_norms[j]
                )

                # Update residual efficiently
                residual = (
                    partial_residual
                    - feature * new_weight
                )

                # Update current weight
                weights[j] = new_weight

            # Calculate prediction error
            mse = np.mean(residual ** 2)

            # Calculate L1 penalty
            penalty = (
                self.alpha / n_samples
                * np.sum(np.abs(weights))
            )

            # Calculate regularized loss
            loss = mse + penalty

            if not np.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss."
                )

            self.loss_history.append(loss)
            self.n_iterations = iteration + 1

            # Check optimality conditions
            gradient = (
                -2 / n_samples
                * (X_centered.T @ residual)
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

            # Stop when optimality conditions are met
            if np.max(violation) <= self.tolerance:

                self.converged = True
                break

        # Store learned weights
        self.weights = weights

        # Recover unregularized intercept
        self.bias = y_mean - X_mean @ weights

        return self


    def predict(self, X):

        if self.weights is None:
            raise ValueError("Train the model first.")

        X = np.asarray(X, dtype=float)

        # Support a single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != len(self.weights):
            raise ValueError("Incorrect feature count.")

        if not np.all(np.isfinite(X)):
            raise ValueError("Invalid feature values.")

        return X @ self.weights + self.bias


    def get_selected_features(self, tolerance=1e-8):
        """
        Return indices of non-zero coefficients.
        """

        if self.weights is None:
            raise ValueError("Train the model first.")

        return np.flatnonzero(
            np.abs(self.weights) > tolerance
        )


# ==========================================
# Example usage
# ==========================================

if __name__ == "__main__":

    # 1. Create dataset
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

    # 2. Initialize model
    model = LassoRegression(alpha=10.0)

    # 3. Train model
    model.fit(X, y)

    # 4. Generate predictions
    y_pred = model.predict(X)

    # 5. Calculate evaluation metrics
    mse = np.mean((y_pred - y) ** 2)

    penalty = (
        model.alpha / len(y)
        * np.sum(np.abs(model.weights))
    )

    # 6. Display results
    print("Weights:", np.round(model.weights, 4))
    print("Bias:", round(model.bias, 4))

    print("Actual values:", y)
    print("Predictions:", np.round(y_pred, 4))

    print("MSE:", round(mse, 4))
    print("L1 Penalty:", round(penalty, 4))
    print("Total Loss:", round(mse + penalty, 4))

    print("Iterations:", model.n_iterations)
    print("Converged:", model.converged)

    print(
        "Selected features:",
        model.get_selected_features()
    )

    # 7. Predict an unseen sample
    prediction = model.predict([[3, 2]])

    print("New prediction:", prediction[0])
