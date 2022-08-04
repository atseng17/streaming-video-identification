import sys
import os
import logging
import datetime
from pytz import timezone

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay


tz = timezone("EST")
logger = logging.getLogger(__name__)
streamHandler = logging.StreamHandler(sys.stdout)
logger.addHandler(streamHandler)
file_handler = logging.FileHandler("log.txt")
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)


def sfl_defaults():
    """Plotting Defaults"""
    plt.style.use("classic")
    plt.rcParams["figure.figsize"] = [8.0, 5.0]
    plt.rcParams["figure.facecolor"] = "w"

    # text size
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["legend.fontsize"] = 12

    # grids
    plt.rcParams["grid.color"] = "k"
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5


def plot_auc(labels, pred, plot_save_dir=None):
    """Helper function for plotting ROC curve
    Args:
        labels (list): ground truth label list
        preds (list): prediction list
    """
    timestamp = str(datetime.datetime.now(tz)).replace(" ", "_")
    timestamp = timestamp.replace(":", "_")
    timestamp = timestamp.replace(".", "_")
    fpr, tpr, thresholds = roc_curve(labels, pred)
    roc_auc = auc(fpr, tpr)
    sfl_defaults()
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ResNet50 model"
    ).plot(ax=ax)
    ax.set_title("ROC curve for Forensic Justice\n\n")
    ax.set_xlabel("\nFalse Positive Rate")
    ax.set_ylabel("True Positive Rate ")
    fig.tight_layout()
    plt.savefig(os.path.join(plot_save_dir,
                f"single_show_auc_curve_{timestamp}.png"))
    plt.close()


def log_single_show_classification_metric(labels, preds, expected_show
                                          ):
    """Get precision, recall, tn, fp, fn, tp
    Args:
        labels (list): a list of gt lables(int)
        preds (list): a list of predicted lables(int)
        expected_show (str): expected show name
    Returns:
        precision (float): precision
        recall (float): precision
        tn (int): true negatives
        fp (int): false positives
        fn (int): false negatives
        tp (int): ture positives
        None
    """
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    precision, recall, fscore, support = score(labels, preds, average="binary")
    logger.info(
        f"Expected Show: {expected_show} >>> tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp},precision:{precision}, recall:{recall}")
    fpr, tpr, thresholds = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    sfl_defaults()
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="Frame Feature"
    ).plot(ax=ax)
    ax.set_title(f"ROC curve for {expected_show}\n\n")
    ax.set_xlabel("\nFalse Positive Rate")
    ax.set_ylabel("True Positive Rate ")
    fig.tight_layout()
    plt.savefig(
        f"single_show_roc_curve_{expected_show}.png")
    plt.close()

    return precision, recall, tn, fp, fn, tp
