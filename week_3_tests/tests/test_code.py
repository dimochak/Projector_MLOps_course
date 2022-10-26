import numpy as np
from sklearn.metrics import accuracy_score


def test_accuracy_correctness():
    preds_1 = np.ones(10)
    preds_2 = np.concatenate((np.ones(5), np.zeros(5)), axis=0)

    true_vals = np.ones(10)

    assert accuracy_score(preds_1, true_vals) == 1.0
    assert accuracy_score(preds_2, true_vals) == 0.5
