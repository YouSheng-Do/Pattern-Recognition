from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

train_df = pd.read_csv('./train.csv')
train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
train_y = train_df["Performance Index"].to_numpy()


# 創建一個線性回歸模型實例
model = LinearRegression()

# 擬合模型
model.fit(train_x, train_y)

# 獲取截距項（beta_0）和斜率（beta_1）
beta_0 = model.intercept_
beta_1 = model.coef_

print(beta_0)
print(beta_1)