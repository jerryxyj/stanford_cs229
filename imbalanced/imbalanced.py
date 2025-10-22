import numpy as np
from src.imbalanced import util
from random import random
from src.linearclass import logreg
import os

### NOTE : You need to complete logreg implementation first!

from src.linearclass.logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1


def _calculate_metrics(y_true, y_pred):
    """Calculates accuracy, balanced accuracy, and class accuracies manually."""
    # True Positives (TP): correctly predicted positive
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # True Negatives (TN): correctly predicted negative
    tn = np.sum((y_true == 0) & (y_pred == 0))
    # False Positives (FP): incorrectly predicted positive
    fp = np.sum((y_true == 0) & (y_pred == 1))
    # False Negatives (FN): incorrectly predicted negative
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate metrics, handling division by zero
    A = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    A0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    A1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    A_bar = 0.5 * (A0 + A1)

    return A, A_bar, A0, A1

def main(train_path, validation_path, plot_path, save_path):
    """Problem: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        plot_path: Path to save plot.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')
    output_plot_path_native = plot_path.replace(WILDCARD, 'naive')
    output_plot_path_upsampling = plot_path.replace(WILDCARD, 'upsampling')
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_val, y_val = util.load_dataset(validation_path, add_intercept=False)

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt() as a 1D numpy array
    vanilla_model = LogisticRegression()
    vanilla_model.fit(x_train, y_train)

    # Evaluate the classifier on the validation dataset
    print("--- Vanilla Logistic Regression Results ---")
    y_pred_vanilla = vanilla_model.predict(x_val)
    A_vanilla, A_bar_vanilla, A0_vanilla, A1_vanilla = _calculate_metrics(y_val, y_pred_vanilla)

    print(f"Overall Accuracy (A): {A_vanilla:.4f}")
    print(f"Balanced Accuracy (Ā): {A_bar_vanilla:.4f}")
    print(f"Class 0 Accuracy (A₀): {A0_vanilla:.4f}")
    print(f"Class 1 Accuracy (A₁): {A1_vanilla:.4f}")

    # Create the plot
    util.plot('Vanilla Logistic Regression Decision Boundary', x_val, y_val, vanilla_model.theta, output_plot_path_native)

    # Save the predictions to output_path_naive
    np.savetxt(output_path_naive, y_pred_vanilla)
    print("----------------------------------------")

    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt() as a 1D numpy array
    # Repeat minority examples 1 / kappa times
    weighted_model = LogisticRegression()
    weighted_model.reweight_fit(x_train, y_train)

    # Evaluate the classifier on the validation dataset
    print("\n--- Re-weighted Logistic Regression Results (from Scratch) ---")
    y_pred_weighted = weighted_model.predict(x_val)
    metrics = _calculate_metrics(y_val, y_pred_weighted)
    print(f"Overall Accuracy (A): {metrics[0]:.4f}")
    print(f"Balanced Accuracy (Ā): {metrics[1]:.4f}")
    print(f"Class 0 Accuracy (A₀): {metrics[2]:.4f}")
    print(f"Class 1 Accuracy (A₁): {metrics[3]:.4f}")

    # Create the plot
    util.plot('Re-weighted Logistic Regression Decision Boundary', x_val, y_val, weighted_model.theta, output_plot_path_upsampling)
    # Save the predictions to output_path_naive
    np.savetxt(output_path_upsampling, y_pred_weighted)
    print("----------------------------------------")


    # *** END CODE HERE

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main(train_path=os.path.join(script_dir, 'train.csv'),
         validation_path=os.path.join(script_dir, 'validation.csv'),
         plot_path=os.path.join(script_dir, 'imbalanced_X_plot.jpg'),
         save_path=os.path.join(script_dir, 'imbalanced_X_pred.txt'))
