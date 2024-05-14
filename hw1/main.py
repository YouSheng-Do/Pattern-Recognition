import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        # print(X.shape)
        all_ones = np.ones(X.shape[0])
        # print(all_ones)
        X = np.insert(X, 0, values=all_ones, axis=1)
        # print(X.shape)
        beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        self.weights = beta[1:]
        self.intercept = beta[0]

    def predict(self, X):
        return self.intercept + np.dot(self.weights, X.T)


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        self.learing_rate = learning_rate
        batchsize = 32
        lambda_ = 0.01  # 0.01 used for L1 regression
        self.weights = np.random.uniform(size=(1, X.shape[1]))
        # print(self.weights)
        self.intercept = 0
        losses = []
        for epoch in range(epochs):
            loss = 0.0
            for i in range(0, X.shape[0], batchsize):
                X_batch = X[i:i + batchsize].T
                y_batch = y[i:i + batchsize].T
                # print(y_batch.shape)
                # y_batch = y_batch.reshape(1, y_batch.shape[0])
                # print(X_batch.shape)
                # print(y_batch.shape)
                # print(self.weights.shape)
                pred_y = np.dot(self.weights, X_batch) + self.intercept
                # print(pred_y.shape)
                error = pred_y - y_batch
                # print(error.shape)
                self.dW = 1 / batchsize * np.dot(error, X_batch.T)
                # print(self.dW.shape)
                self.db = 1 / batchsize * np.sum(error)
                loss = loss + compute_mse(pred_y, y_batch) + lambda_ * np.sum(np.sign(self.weights))
                # print(loss)
                # print(np.sign(self.weights).shape)
                self.weights = self.weights - learning_rate * self.dW - learning_rate * lambda_ * np.sign(self.weights)
                self.intercept = self.intercept - learning_rate * self.db
            loss /= (X.shape[0] / batchsize)
            if epoch % 10000 == 0:
                logger.info(f'EPOCH {epoch}, loss={loss}')
            losses.append(loss)
        # print(losses)
        self.weights = np.squeeze(self.weights)
        return losses

    def predict(self, X):
        pred_y = self.intercept + np.dot(self.weights, X.T)
        return pred_y.T

    def plot_learning_curve(self, losses):
        plt.plot(np.squeeze(losses))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('learning rate = ' + str(self.learing_rate))
        plt.show()
        # plt.savefig("learning_curve.png")


def compute_mse(prediction, ground_truth):
    return np.mean(np.square(prediction - ground_truth))


def main():
    # np.random.seed(1)
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    train_y = train_y.reshape(train_y.shape[0], 1)
    losses = LR_GD.fit(train_x, train_y, learning_rate=3e-5, epochs=30000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    # print(y_preds_cf)
    y_preds_gd = LR_GD.predict(test_x)
    # print(y_preds_gd)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).sum()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = ((mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
