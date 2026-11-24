import numpy as np


class LogisticRegression:
    """
    Binary Logistic Regression implemented from scratch
    using mathematics and NumPy.

    Optimization: Batch Gradient Descent
    Loss: Binary Cross-Entropy
    Regularization: Optional L2
    """

    def __init__(
        self,
        learning_rate=0.1,
        epochs=5000,
        alpha=0.0,
        tolerance=1e-8,
        threshold=0.5
    ):

        # Validate hyperparameters
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive and finite."
            )

        if (
            not isinstance(epochs, (int, np.integer))
            or isinstance(epochs, (bool, np.bool_))
            or epochs <= 0
        ):
            raise ValueError(
                "epochs must be a positive integer."
            )

        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError(
                "alpha must be non-negative and finite."
            )

        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(
                "tolerance must be positive and finite."
            )

        if not 0 < threshold < 1:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        # Hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.tolerance = tolerance
        self.threshold = threshold

        # Learnable parameters
        self.weights = None
        self.bias = None

        # Training statistics
        self.loss_history = []
        self.n_iterations = 0
        self.converged = False


    @staticmethod
    def sigmoid(z):
        """
        Numerically stable sigmoid activation.

        Avoids exponential overflow for large
        positive and negative inputs.
        """

        z = np.asarray(z, dtype=float)

        a = np.exp(-np.abs(z))

        return np.where(
            z >= 0,
            1 / (1 + a),
            a / (1 + a)
        )


    def _loss(self, X, y, w, b):
        """
        Calculate Binary Cross-Entropy directly
        from logits, including the L2 penalty.
        """

        # Calculate logits
        z = X @ w + b

        # Numerically stable Binary Cross-Entropy
        bce = np.mean(
            np.logaddexp(
                0.0,
                (1 - 2 * y) * z
            )
        )

        # L2 regularization penalty
        penalty = (
            self.alpha / (2 * len(y))
            * np.dot(w, w)
        )

        return bce + penalty


    def _gradients(self, X, y, w, b):
        """
        Calculate weight and bias gradients.
        """

        # Calculate logits
        z = X @ w + b

        # Numerically stable prediction error
        # Mathematically equivalent to sigmoid(z) - y
        error = np.where(
            y == 1,
            -self.sigmoid(-z),
            self.sigmoid(z)
        )

        # Weight gradient with L2 regularization
        dw = (
            X.T @ error + self.alpha * w
        ) / len(y)

        # Bias gradient
        db = np.mean(error)

        return dw, db


    def fit(self, X, y):
        """
        Train the Logistic Regression model.
        """

        # Convert inputs into NumPy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Support a 1D single-feature dataset
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Validate input dimensions
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError(
                "X must be 1D or 2D; y must be 1D."
            )

        # Validate dataset size
        if X.shape[0] != y.size or X.size == 0:
            raise ValueError(
                "X and y must contain matching, "
                "nonempty samples."
            )

        # Validate numerical values
        if (
            not np.all(np.isfinite(X))
            or not np.all(np.isfinite(y))
        ):
            raise ValueError(
                "X and y must contain finite values."
            )

        # Ensure both binary classes are present
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError(
                "Training labels must contain both 0 and 1."
            )

        # Step 1: Initialize parameters
        w = np.zeros(X.shape[1], dtype=float)
        b = 0.0

        # Step 2: Record initial loss
        history = [
            self._loss(X, y, w, b)
        ]

        # ==========================================
        # Batch Gradient Descent Training Loop
        # ==========================================

        for _ in range(self.epochs):

            # Step 3: Calculate gradients
            dw, db = self._gradients(
                X, y, w, b
            )

            # Step 4: Check convergence
            gradient_size = max(
                np.max(np.abs(dw)),
                abs(db)
            )

            if gradient_size <= self.tolerance:
                break

            # Step 5: Update parameters
            w = w - self.learning_rate * dw

            b = b - self.learning_rate * db

            # Step 6: Calculate updated loss
            loss = self._loss(X, y, w, b)

            # Detect numerical divergence
            if not np.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite loss. Scale X "
                    "or reduce learning_rate."
                )

            # Step 7: Store training loss
            history.append(loss)

        # Check convergence using final parameters
        dw, db = self._gradients(X, y, w, b)

        self.converged = bool(
            max(
                np.max(np.abs(dw)),
                abs(db)
            ) <= self.tolerance
        )

        # Store learned parameters
        self.weights = w
        self.bias = float(b)

        # Store training statistics
        self.loss_history = history
        self.n_iterations = len(history) - 1

        return self


    def decision_function(self, X):
        """
        Return the raw logits before sigmoid.
        """

        if self.weights is None:
            raise ValueError("Fit the model first.")

        X = np.asarray(X, dtype=float)

        # Handle a single scalar input
        if X.ndim == 0:
            X = X.reshape(1, 1)

        # Handle one-dimensional input
        elif X.ndim == 1:

            if self.weights.size == 1:

                # Multiple samples with one feature
                X = X.reshape(-1, 1)

            else:

                # One sample with multiple features
                X = X.reshape(1, -1)

        # Validate feature count
        if (
            X.ndim != 2
            or X.shape[1] != self.weights.size
        ):
            raise ValueError(
                "Incorrect feature count."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X must contain finite values."
            )

        return X @ self.weights + self.bias


    def predict_proba(self, X):
        """
        Return probabilities for both classes.

        Column 0: P(y = 0)
        Column 1: P(y = 1)
        """

        # Calculate logits
        z = self.decision_function(X)

        # Class 0 probability
        p0 = self.sigmoid(-z)

        # Class 1 probability
        p1 = self.sigmoid(z)

        return np.column_stack((p0, p1))


    def predict(self, X, threshold=None):
        """
        Convert probabilities into binary labels.
        """

        t = (
            self.threshold
            if threshold is None
            else threshold
        )

        if not 0 < t < 1:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        # Positive class probabilities
        probabilities = self.predict_proba(X)[:, 1]

        # Convert probabilities into labels
        return (probabilities >= t).astype(int)


    def score(self, X, y):
        """
        Calculate classification accuracy.
        """

        y = np.asarray(y)

        predictions = self.predict(X)

        if (
            y.shape != predictions.shape
            or y.size == 0
            or not np.all(np.isin(y, [0, 1]))
        ):
            raise ValueError(
                "Invalid binary targets."
            )

        return float(
            np.mean(y == predictions)
        )


# ==========================================
# Classification Evaluation Metrics
# ==========================================

def classification_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Validate input shapes
    if (
        y_true.ndim != 1
        or y_true.shape != y_pred.shape
        or y_true.size == 0
    ):
        raise ValueError(
            "Expected matching nonempty 1D labels."
        )

    # Validate binary labels
    if (
        not np.all(np.isin(y_true, [0, 1]))
        or not np.all(np.isin(y_pred, [0, 1]))
    ):
        raise ValueError(
            "Labels must be 0 or 1."
        )

    # True Negatives
    tn = int(np.sum(
        (y_true == 0) & (y_pred == 0)
    ))

    # False Positives
    fp = int(np.sum(
        (y_true == 0) & (y_pred == 1)
    ))

    # False Negatives
    fn = int(np.sum(
        (y_true == 1) & (y_pred == 0)
    ))

    # True Positives
    tp = int(np.sum(
        (y_true == 1) & (y_pred == 1)
    ))

    # Prevent division by zero
    def safe_divide(a, b):

        return a / b if b else 0.0

    return {

        "confusion_matrix": np.array([
            [tn, fp],
            [fn, tp]
        ]),

        "accuracy": safe_divide(
            tp + tn,
            len(y_true)
        ),

        "precision": safe_divide(
            tp,
            tp + fp
        ),

        "recall": safe_divide(
            tp,
            tp + fn
        ),

        "specificity": safe_divide(
            tn,
            tn + fp
        ),

        "f1": safe_divide(
            2 * tp,
            2 * tp + fp + fn
        )
    }


# ==========================================
# Main Program
# ==========================================

if __name__ == "__main__":

    # 1. Create dataset
    X = np.array([
        -4, -3, -2, -1,
         1,  2,  3,  4
    ], dtype=float)

    y = np.array([
        0, 0, 0, 1,
        0, 1, 1, 1
    ])

    # 2. Initialize Logistic Regression
    model = LogisticRegression(
        learning_rate=0.1,
        epochs=5000,
        alpha=0.0,
        tolerance=1e-8,
        threshold=0.5
    )

    # 3. Train the model
    model.fit(X, y)

    # 4. Generate probabilities
    probabilities = model.predict_proba(X)

    # 5. Generate class predictions
    predictions = model.predict(X)

    # 6. Calculate classification metrics
    metrics = classification_metrics(
        y,
        predictions
    )

    # 7. Display model parameters
    print("MODEL PARAMETERS")
    print("----------------")

    print(
        "Weights:",
        np.round(model.weights, 4)
    )

    print(f"Bias: {model.bias:.4f}")

    # 8. Display training statistics
    print("\nTRAINING STATISTICS")
    print("-------------------")

    print(
        f"Initial loss: {model.loss_history[0]:.6f}"
    )

    print(
        f"Final loss: {model.loss_history[-1]:.6f}"
    )

    print("Iterations:", model.n_iterations)

    print("Converged:", model.converged)

    # 9. Display predictions
    print("\nPREDICTIONS")
    print("-----------")

    print("Actual labels:", y)

    print("Predicted labels:", predictions)

    print(
        "Probabilities:",
        np.round(probabilities[:, 1], 4)
    )

    # 10. Display evaluation metrics
    print("\nMODEL EVALUATION")
    print("----------------")

    print(
        "Confusion matrix:\n",
        metrics["confusion_matrix"]
    )

    for name in [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1"
    ]:

        print(f"{name.capitalize()}: {metrics[name]:.4f}")

    # 11. Predict unseen input
    X_new = 5

    new_probability = model.predict_proba(X_new)

    new_prediction = model.predict(X_new)

    print("\nNEW PREDICTION")
    print("--------------")

    print("Input:", X_new)

    print(
        f"Probability of class 1: "
        f"{new_probability[0, 1]:.4f}"
    )

    print("Predicted class:", new_prediction[0])
