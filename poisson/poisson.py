import numpy as np
from src.poisson import util
import matplotlib.pyplot as plt
import os


def main(lr, train_path, eval_path, plot_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        plot_path: Path to save plot of loss vs. iteration number, or None to not save.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path as a 1D numpy array
    model = PoissonRegression()
    model.fit(x_train, y_train)
    prediction = model.predict(x_val)
    util.plot(prediction, y_val, model.theta, plot_path)
    np.savetxt(save_path, prediction)
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Update the parameter by step_size * (sum of the gradient over examples)

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Initialize self.theta to zero vector if it is None
        # Implement gradient ascent loop with convergence check
        x = util.add_intercept(x)
        n_samples, n_features = x.shape
        self.theta = np.zeros(n_features) if self.theta is None else self.theta

        delta_norm = float('inf')

        while delta_norm >= 1e-5:
            theta_old = self.theta.copy()

            # Calculate predictions (lambda) for the entire batch
            lambda_vals = np.exp(x @ self.theta)

            # Calculate the gradient for the entire batch
            gradient = x.T @ (y - lambda_vals)

            # Update theta using the gradient ascent rule
            self.theta += self.step_size * gradient

            # Calculate the norm of the change in theta for the stopping criterion
            delta_norm = np.linalg.norm(self.theta - theta_old)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        x = util.add_intercept(x)
        lambda_vals = np.exp(x @ self.theta)
        return lambda_vals
        # *** END CODE HERE ***

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main(lr=1e-5,
        train_path=os.path.join(script_dir, 'train.csv'),
        eval_path=os.path.join(script_dir, 'valid.csv'),
        plot_path=os.path.join(script_dir, 'possion_plot.jpg'),
        save_path=os.path.join(script_dir, 'poisson_pred.txt'))
