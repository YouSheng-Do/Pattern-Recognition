import typing as t
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layer model.
    No non-linear activation allowed.
    """
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        
        self.layers = nn.Sequential(
                        # nn.Linear(input_dim, input_dim),
                        nn.Linear(input_dim, 1),
                        nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def entropy_loss(outputs, targets):
    return - torch.mean((targets * torch.log(outputs + 1e-20) + \
                                                        (1 - targets) * torch.log(1 - outputs + 1e-20)))


def plot_learners_roc(
    X_test,
    clf,
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    X_test = torch.from_numpy(X_test).float()
    plt.figure(figsize=(5, 5))
    for model in clf.learners:
        output = model(X_test).detach().numpy()
        fpr, tpr, thresholds = roc_curve(y_trues, output)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC={auc_score:.4f}')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(fpath)

def plot_feature_importance(
    feature_names,
    feature_importance,
    save_path
):
    plt.figure(figsize=(5, 5))
    plt.barh(feature_names, feature_importance)
    plt.savefig(save_path)