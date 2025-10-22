import numpy as np
from src.linearclass import util
import os

def main(train_path, valid_path, plot_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        plot_path: Path to save plot of decision boundary.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_val, y_val = util.load_dataset(valid_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    model = LogisticRegression()
    model.fit(x_train, y_train)
    prediction = model.predict(x_val)
    # Plot decision boundary on top of validation set set
    util.plot(x_val, y_val, model.theta, plot_path, correction=1.0)
    # Use np.savetxt to save predictions on eval set to save_path as a 1D numpy array
    np.savetxt(save_path, prediction)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose


    def _sigmoid(self, z):
        # normally sigmoid should be 1 / (1 + np.exp(-z)), but if z is too large, it will overflow.
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Add an intercept term to x
        n, d = x.shape
        x_intercept = util.add_intercept(x)

        # Initialize theta to a vector of zeros if not provided
        if self.theta is None:
            self.theta = np.zeros(d + 1)

        for i in range(self.max_iter):
            theta_old = np.copy(self.theta)

            # Calculate the hypothesis (predictions)
            z = x_intercept @ self.theta
            h = self._sigmoid(z)

            # Calculate the gradient of the negative log-likelihood
            grad = x_intercept.T @ (h - y)

            # Calculate the Hessian matrix
            # W is a diagonal matrix of weights
            W = np.diag(h * (1 - h))
            H = x_intercept.T @ W @ x_intercept

            # Add a small regularization term for numerical stability
            H_inv = np.linalg.inv(H + 1e-8 * np.eye(H.shape[0]))

            # Update theta using Newton's step
            self.theta = self.theta - H_inv @ grad

            # Check for convergence using the L1 norm of the change in theta
            l1_norm_diff = np.linalg.norm(self.theta - theta_old, ord=1)

            if self.verbose and i % 10 == 0:
                print(f'Iteration {i}: L1 norm of delta_theta = {l1_norm_diff:.6f}')

            if l1_norm_diff < self.eps:
                print(f'Converged after {i + 1} iterations.')
                break
        # *** END CODE HERE ***


    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Add an intercept term to x
        x = util.add_intercept(x)
        probabilities = self._sigmoid(x @ self.theta)
        return (probabilities >= 0.5).astype(int)
        # *** END CODE HERE ***

    def reweight_fit(self, x, y):
        """Run gradient descent to minimize J(theta) for logistic regression. This is only for imbalanced question"""

        x = util.add_intercept(x)
        n_samples, n_features = x.shape
        self.theta = np.zeros(n_features)

        weights = np.ones(n_samples)
        n_class0 = np.sum(y == 0)
        n_class1 = np.sum(y == 1)
        weights[y == 0] = n_samples / (2 * n_class0)
        weights[y == 1] = n_samples / (2 * n_class1)

        for _ in range(self.max_iter):
            z = x @ self.theta
            h = self._sigmoid(z)
            error = y - h
            weighted_error = weights * error
            gradient = x.T @ weighted_error
            self.theta += self.step_size * gradient

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    main(train_path=os.path.join(script_dir, 'ds1_train.csv'),
         valid_path=os.path.join(script_dir, 'ds1_valid.csv'),
         plot_path=os.path.join(script_dir, 'logreg_plot_1.jpg'),
         save_path=os.path.join(script_dir, 'logreg_pred_1.txt'))

    main(train_path=os.path.join(script_dir, 'ds2_train.csv'),
         valid_path=os.path.join(script_dir, 'ds2_valid.csv'),
         plot_path=os.path.join(script_dir, 'logreg_plot_2.jpg'),
         save_path=os.path.join(script_dir, 'logreg_pred_2.txt'))