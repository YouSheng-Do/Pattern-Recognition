import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import roc_auc_score


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        n_samples, n_features = inputs.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0
        batch_size = 32
        lambda_ = 0.001
        self.losses = []

        for i in range(self.num_iterations):
            permutation = np.random.permutation(n_samples)
            inputs_shuffled = inputs[permutation]
            targets_shuffled = targets[permutation]
            loss = 0.0
            for j in range(0, n_samples, batch_size):
                inputs_batch = inputs_shuffled[j:j + batch_size]
                targets_batch = targets_shuffled[j:j + batch_size]
                predicts = self.sigmoid(inputs_batch.dot(self.weights) + self.intercept)
                errors = predicts - targets_batch
                loss += np.mean(-targets_batch * np.log(predicts) - (1 - targets_batch) * np.log(1 - predicts))
                loss += lambda_ * np.sum(np.sign(self.weights))
                self.weights -= self.learning_rate * inputs_batch.T.dot(errors) / batch_size
                self.weights -= self.learning_rate * lambda_ * np.sign(self.weights)
                self.intercept -= self.learning_rate * np.mean(errors)
            self.losses.append(loss)
            # print(loss)

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        pred_probs = self.sigmoid(inputs.dot(self.weights) + self.intercept)
        pred_classes = (pred_probs >= 0.5).astype(int)

        return pred_probs, pred_classes

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))


class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        inputs_class_0 = inputs[targets == 0]
        inputs_class_1 = inputs[targets == 1]
        self.m0 = np.mean(inputs_class_0, axis=0)
        self.m1 = np.mean(inputs_class_1, axis=0)
        # print(self.m0)
        self.sb = np.outer((self.m1 - self.m0), (self.m1 - self.m0))
        # self.sb = np.dot((self.m1 - self.m0).reshape(2, 1), (self.m1 - self.m0).reshape(1, 2))
        # print(self.sb)
        self.sw = np.dot((inputs_class_0 - self.m0).T, (inputs_class_0 - self.m0)) + \
            np.dot((inputs_class_1 - self.m1).T, (inputs_class_1 - self.m1))
        # print(self.sw)
        eigenvalues, eigenvectors = np.linalg.eigh(np.dot((self.sb), np.linalg.inv(self.sw)))
        self.w = eigenvectors[np.argmax(eigenvalues)]
        # self.w = np.dot((self.m1 - self.m0), np.linalg.inv(self.sw))
        # print(self.w)

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Sequence[t.Union[int, bool]]:
        projections = inputs.dot(self.w.T)
        # print(projections)

        mean_projection_class_0 = np.dot(self.m0, self.w.T)
        mean_projection_class_1 = np.dot(self.m1, self.w.T)
        threshold = (mean_projection_class_0 + mean_projection_class_1) / 2
        # print(threshold)

        preds_class = (projections >= threshold).astype(int)
        # print(preds_class)

        return preds_class

    def plot_projection(self, inputs: npt.NDArray[float]):
        predictions = self.predict(inputs)
        projections = inputs.dot(self.w.T)

        mean_projection_class_0 = np.dot(self.m0, self.w)
        mean_projection_class_1 = np.dot(self.m1, self.w)
        self.slope = self.w[1] / self.w[0]
        bias = -(mean_projection_class_0 + mean_projection_class_1) / 2
        title = f"Projection Line: w={self.slope:.6f}, b={bias:.6f}"

        plt.figure(figsize=(5, 5))
        plt.title(title)

        x0 = inputs[:, 0]
        x1 = inputs[:, 1]
        proj_x0 = projections / self.w.dot(self.w) * self.w[0]
        proj_x1 = projections / self.w.dot(self.w) * self.w[1]

        plt.plot(proj_x0, proj_x1, 'k', label='Projection Line')
        plt.scatter(x0, x1, c=predictions, cmap='bwr')
        plt.scatter(proj_x0, proj_x1, c=predictions, cmap='bwr')

        for i in range(len(x0)):
            plt.plot([x0[i], proj_x0[i]], [x1[i], proj_x1[i]], 'b-', alpha=0.2)

        plt.gca().set_aspect('equal')
        plt.show()


def compute_auc(y_trues, y_preds) -> float:
    return roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds) -> float:
    n_correct = np.sum(y_preds == y_trues)
    accuracy = n_correct / y_preds.shape[0]
    return accuracy


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=1e-1,  # You can modify the parameters as you want
        num_iterations=200,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    FLD_.fit(x_train, y_train)
    y_preds = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_preds)
    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')
    FLD_.plot_projection(x_test)


if __name__ == '__main__':
    main()
