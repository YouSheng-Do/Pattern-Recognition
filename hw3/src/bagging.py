import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """Implement your code here"""

        losses_of_models = []
        np.random.seed(2)
        for model in self.learners:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            sample_idx = [np.random.randint(0, X_train.shape[0]) for _ in range(X_train.shape[0])]
            X_bagging = torch.from_numpy(X_train[sample_idx]).float()
            y_bagging = torch.from_numpy(y_train[sample_idx]).float()
            
            model.train()
            for _ in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(X_bagging)
                outputs = outputs.squeeze()
                loss = entropy_loss(outputs, y_bagging)
                
                loss.backward()
                optimizer.step()
            
            losses_of_models.append(loss.item())
                
        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        X = torch.from_numpy(X).float()
        outputs = torch.zeros(X.shape[0])
        for model in self.learners:
            output = model(X).squeeze()
            outputs += (output >= 0.5).int()
            
        # print(outputs)
        return (outputs >= len(self.learners) / 2).int().numpy(), outputs.numpy()

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""
        feature_importance = np.zeros(6)
        # print(feature_importance)
        for model in self.learners:
            feature_importance += (model.layers[0].weight.abs()).detach().numpy().squeeze()
        return feature_importance

