import matplotlib.pyplot as plt
import numpy as np

def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x

def load_dataset(csv_path, add_intercept=True):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i].startswith('y')]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def plot(title, x_val, y_val, theta, save_path):
    """Plot dataset and fitted logistic regression parameters.

    Args:
        title: Title for plot.
        x_val: Matrix of training examples, one per row.
        y_val: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
    """
    # Plot dataset
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.scatter(x_val[y_val == 0][:, 0], x_val[y_val == 0][:, 1], marker='o', label='Class 0')
    plt.scatter(x_val[y_val == 1][:, 0], x_val[y_val == 1][:, 1], marker='x', label='Class 1')
    x1_vals = np.array([x_val[:, 0].min(), x_val[:, 0].max()])
    x2_vals = -(theta[0] + theta[1] * x1_vals) / theta[2]
    plt.plot(x1_vals, x2_vals, color='red', label='Decision Boundary')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.legend()
    plt.savefig(save_path)
