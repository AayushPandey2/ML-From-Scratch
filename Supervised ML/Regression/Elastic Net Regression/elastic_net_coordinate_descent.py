import numpy as np


def soft_threshold(z, threshold):
    """
    Shrink values towards zero using
    the soft-thresholding operator.
    """

    return np.sign(z) * np.maximum(
        np.abs(z) - threshold,
        0.0
    )


class ElasticNetRegression:

    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.5,
        max_iterations=10000,
        tolerance=1e-9
    ):

        # Validate hyperparameters
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("Invalid alpha.")

        if not 0 <= l1_ratio <= 1:
            raise ValueError("L1 ratio must be between 0 and 1.")

        if (
            not isinstance(max_iterations, int)
            or max_iterations <= 0
        ):
            raise ValueError("Invalid iteration count.")

        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Invalid tolerance.")

        # Hyperparameters
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Calculate regularization strengths
        self.lambda1 = alpha * l1_ratio
        self.lambda2 = alpha * (1 - l1_ratio)

        # Learnable parameters
        self.weights = None
        self.bias = None

        # Training statistics
        self.loss_history = []
        self.mse_history = []
        self.l1_history = []
        self.l2_history = []

        self.converged = False
        self.n_iterations = 0


    def fit(self, X, y):
        """
        Train Elastic Net Regression using
        Coordinate Descent.
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

        # Step 1: Extract dimensions
        n_samples, n_features = X.shape

        # Step 2: Calculate means
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)

        # Step 3: Center dataset
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Step 4: Initialize weights
        weights = np.zeros(n_features)

        # Step 5: Calculate squared feature norms
        feature_norms = np.sum(
            X_centered ** 2,
            axis=0
        )

        # Step 6: Initialize residual
        residual = y_centered.copy()

        # Reset training history
        initial_mse = np.mean(residual ** 2)

        self.loss_history = [initial_mse]
        self.mse_history = [initial_mse]
        self.l1_history = [0.0]
        self.l2_history = [0.0]

        self.converged = False
        self.n_iterations = 0

        # ======================================
        # Coordinate Descent Training Loop
        # ======================================

        for iteration in range(self.max_iterations):

            for j in range(n_features):

                # Extract current feature
                feature = X_centered[:, j]

                # Calculate regularized denominator
                denominator = (
                    feature_norms[j]
                    + self.lambda2
                )

                if denominator == 0:
                    continue

                # Remove current feature contribution
                partial_residual = (
                    residual
                    + feature * weights[j]
                )

                # Calculate rho
                rho = feature @ partial_residual

                # Apply L1 and L2 regularization
                new_weight = (
                    soft_threshold(
                        rho,
                        self.lambda1 / 2
                    )
                    / denominator
                )

                # Update residual
                residual = (
                    partial_residual
                    - feature * new_weight
                )

                # Update current coefficient
                weights[j] = new_weight

            # Calculate prediction error
            mse = np.mean(residual ** 2)

            # Calculate L1 penalty
            l1_penalty = (
                self.lambda1 / n_samples
                * np.sum(np.abs(weights))
            )

            # Calculate L2 penalty
            l2_penalty = (
                self.lambda2 / n_samples
                * np.sum(weights ** 2)
            )

            # Calculate total objective
            loss = mse + l1_penalty + l2_penalty

            if not np.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss."
                )

            # Store training history
            self.mse_history.append(mse)
            self.l1_history.append(l1_penalty)
            self.l2_history.append(l2_penalty)
            self.loss_history.append(loss)

            self.n_iterations = iteration + 1

            # Calculate gradient of smooth objective
            gradient = (
                -2 / n_samples
                * (X_centered.T @ residual)
                + 2 * self.lambda2 / n_samples
                * weights
            )

            l1_strength = self.lambda1 / n_samples

            # Check optimality conditions
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

            # Convergence check
            if np.max(violation) <= self.tolerance:

                self.converged = True
                break

        # Store learned coefficients
        self.weights = weights

        # Recover unregularized intercept
        self.bias = (
            y_mean - X_mean @ weights
        )

        return self


    def predict(self, X):
        """
        Generate predictions using learned parameters.
        """

        if self.weights is None:
            raise ValueError("Train the model first.")

        X = np.asarray(X, dtype=float)

        # Support prediction for one sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError("Invalid input dimensions.")

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

    # 1. Dataset
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
    model = ElasticNetRegression(
        alpha=12,
        l1_ratio=5 / 6
    )

    # 3. Train model
    model.fit(X, y)

    # 4. Generate predictions
    y_pred = model.predict(X)

    # 5. Display model parameters
    print("Lambda 1:", round(model.lambda1, 4))
    print("Lambda 2:", round(model.lambda2, 4))

    print("Weights:", np.round(model.weights, 4))
    print("Bias:", round(model.bias, 4))

    print("Actual values:", y)
    print("Predictions:", np.round(y_pred, 4))

    # 6. Display training results
    print("MSE:", round(model.mse_history[-1], 4))
    print("L1 Penalty:", round(model.l1_history[-1], 4))
    print("L2 Penalty:", round(model.l2_history[-1], 4))
    print("Total Loss:", round(model.loss_history[-1], 4))

    print("Converged:", model.converged)
    print("Iterations:", model.n_iterations)

    print(
        "Selected features:",
        model.get_selected_features()
    )

    # 7. Predict unseen input
    prediction = model.predict([[3, 2]])

    print("New prediction:", round(prediction[0], 4))
