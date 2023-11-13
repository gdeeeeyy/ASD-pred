import numpy as np
import matplotlib.pyplot as plt

class PrecisionRecallCurveGenerator:
    def __init__(self, y_true):
        self.y_true = np.array(y_true)

    def logistic_scoring_function(self):
        return 1 / (1 + np.exp(-np.random.randn(len(self.y_true))))

    def generate_curve(self):
        y_scores = self.logistic_scoring_function()

        # Sort the scores and true labels in descending order of scores
        sort_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = self.y_true[sort_indices]
        y_scores_sorted = y_scores[sort_indices]

        # Calculate precision and recall at different thresholds
        precision_values = []
        recall_values = []

        for threshold in np.linspace(0, 1, 100):
            y_pred = (y_scores_sorted > threshold).astype(int)
            true_positive = np.sum((y_true_sorted == 1) & (y_pred == 1))
            false_positive = np.sum((y_true_sorted == 0) & (y_pred == 1))
            false_negative = np.sum((y_true_sorted == 1) & (y_pred == 0))

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

            precision_values.append(precision)
            recall_values.append(recall)

        # Plot Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall_values, precision_values, color='blue', lw=2, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
