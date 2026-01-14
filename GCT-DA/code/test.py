import torch
from utils import get_link_labels
from metric import *

@torch.no_grad()
def mytest(model, val_pos_edge_index, val_neg_edge_index, output):
    model.eval()

    val_link_logits = model.decode(output, val_pos_edge_index, val_neg_edge_index)
    val_link_probs = val_link_logits.sigmoid()
    val_link_labels = get_link_labels(val_pos_edge_index, val_neg_edge_index)
    auc, acc, prc, pre, rec, f1, tpr, fpr, recall, precision = get_metrics(val_link_labels, val_link_probs)

    return auc, acc, prc, pre, rec, f1, tpr, fpr, recall, precision
