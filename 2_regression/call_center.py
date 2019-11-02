import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import mlinsights.mlmodel as mi_model


plt.figure(figsize=(20, 8))

call_center_df = pd.read_csv("data/call_center.csv", parse_dates=["timestamp"])

call_center_df.at[17, "calls"] = 500
call_center_df.at[18, "calls"] = 500
call_center_df.at[19, "calls"] = 500

# X = np.array([t.value for t in call_center_df["timestamp"]]).reshape(-1, 1)
X = np.array(call_center_df.index).reshape(-1, 1)
y = np.array(call_center_df["calls"]).reshape(-1, 1)

plt.plot(X, y, color="b")

border_values = np.array([X[0][0], X[-1][0]]).reshape(-1, 1)

print("OLS:")

ols_model = lm.LinearRegression()
ols_model.fit(X, y)

ols_trend = ols_model.predict(border_values)

print("Slope: {}".format(ols_model.coef_[0][0]))
print("Overall change: {}".format(ols_trend[1][0] - ols_trend[0][0]))

plt.plot(border_values, ols_trend, color="r")

print("LAD:")

y = np.array(call_center_df["calls"])

lad_model = mi_model.QuantileLinearRegression()
lad_model.fit(X, y)

lad_trend = lad_model.predict(border_values)

print("Slope: {}".format(lad_model.coef_[0]))
print("Overall change: {}".format(lad_trend[1] - lad_trend[0]))

plt.plot(border_values, lad_trend, color="g")

plt.show()
