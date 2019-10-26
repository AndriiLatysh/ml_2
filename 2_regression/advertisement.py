import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm


sales_df = pd.read_csv("data/advertising.csv", index_col=0)

ad_data = sales_df[["TV", "radio", "newspaper"]]
sales_data = sales_df[["sales"]]

sales_model = lm.LinearRegression()
sales_model.fit(X=ad_data, y=sales_data)

print(sales_model.coef_)
print(sales_model.intercept_)

