import pandas as pd
import sklearn.linear_model as lm


advertising_data = pd.read_csv("data/advertising.csv", index_col=0)
print(advertising_data)

ad_data = advertising_data[["TV", "radio", "newspaper"]]
sales_data = advertising_data[["sales"]]

linear_regression = lm.LinearRegression()
ridge_regression = lm.Ridge()
lasso_regression = lm.Lasso()

linear_regression.fit(ad_data, sales_data)
ridge_regression.fit(ad_data, sales_data)
lasso_regression.fit(ad_data, sales_data)

print("Linear regression:")
print(linear_regression.coef_)
print(linear_regression.intercept_)

print("Ridge regression:")
print(ridge_regression.coef_)
print(ridge_regression.intercept_)

print("Lasso regression:")
print(lasso_regression.coef_)
print(lasso_regression.intercept_)
