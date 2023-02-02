import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)


def eval_clf(y_test, y_pred, disp=0):
    if isinstance(y_test, (pd.core.frame.DataFrame, pd.core.series.Series)):
        y_test = y_test.astype(str)
    if isinstance(y_pred, (pd.core.frame.DataFrame, pd.core.series.Series)):
        y_pred = y_pred.astype(str)
    clf_report = classification_report(y_test,
                                       y_pred, zero_division=0)

    print(clf_report)

    if disp:
        test_labels = set(np.unique(y_test))
        pred_labels = set(np.unique(y_pred))
        labels = sorted(test_labels.union(pred_labels))

        conf_matrix = confusion_matrix(y_test,
                                       y_pred)
        disp = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.grid(False)
        disp.plot(ax=ax)

    return round(f1_score(y_test, y_pred, average='micro'), 2)
