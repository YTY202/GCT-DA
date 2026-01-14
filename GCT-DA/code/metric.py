import numpy as np
from sklearn import metrics

def get_metrics(real_score, predict_score):
    pre_scores = np.squeeze(predict_score.cpu().detach().numpy())
    real_labels = np.squeeze(real_score.cpu().detach().numpy())
    fpr, tpr, thresholds = metrics.roc_curve(real_labels, pre_scores)
    precision, recall, _ = metrics.precision_recall_curve(real_labels, pre_scores)
    auc = metrics.auc(fpr, tpr)
    prc = metrics.auc(recall, precision)

    pred_labels = [0 if j < 0.5 else 1 for j in pre_scores]
    acc = metrics.accuracy_score(real_labels, pred_labels)
    pre = metrics.precision_score(real_labels, pred_labels)
    rec = metrics.recall_score(real_labels, pred_labels)
    f1 = metrics.f1_score(real_labels, pred_labels)

    return auc, acc, prc, pre, rec, f1, tpr, fpr, recall, precision
