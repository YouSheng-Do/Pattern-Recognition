import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        self.sample_weights = None
        # create 10 learners, dont change.
        torch.manual_seed(31)
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001):
        """Implement your code here"""
        losses_of_models = []
        self.sample_weights = torch.from_numpy(np.ones(X_train.shape[0]) / X_train.shape[0])
        
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()

        for model in self.learners:
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
            model.train()
            for _ in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(X_train)
                outputs = outputs.squeeze()
                loss = entropy_loss(outputs, y_train)

                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                pred_labels = (outputs >= 0.5).float().squeeze()
                error_rate = torch.sum(self.sample_weights * (pred_labels != y_train).float()) \
                                        / torch.sum(self.sample_weights)

            alpha = 0.5 * torch.log((1 - error_rate) / (error_rate + 1e-20))
            self.alphas.append(alpha)
            self.sample_weights *= torch.exp(- alpha * y_train * outputs.detach().squeeze())
            self.sample_weights /= torch.sum(self.sample_weights)
            
            losses_of_models.append(loss.item())

        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        outputs = torch.zeros(X.shape[0])
        X = torch.from_numpy(X).float()
        for alpha, model in zip(self.alphas, self.learners):
            output = model(X).detach().squeeze()
            outputs += alpha * (output >= 0.5).float()
        outputs /= np.sum(self.alphas)
        # print(outputs)
        return (outputs >= 0.5).int().numpy(), outputs.numpy()

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""
        feature_importance = np.zeros(6)
        # print(feature_importance)
        for alpha, model in zip(self.alphas, self.learners):
            feature_importance += (model.layers[0].weight.abs() * alpha).detach().numpy().squeeze()
        return feature_importance
