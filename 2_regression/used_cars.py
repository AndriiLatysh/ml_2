import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm


used_cars_df = pd.read_csv("data/true_car_listings.csv")

used_cars_df[["Age"]] = 2019 - used_cars_df[["Year"]]

model_list = used_cars_df[["Model", "Vin"]].groupby("Model").count().sort_values(by="Vin", ascending=False)

selected_model_df = used_cars_df[used_cars_df["Model"] == "Civic"]

plt.scatter(selected_model_df[["Age"]], selected_model_df[["Price"]])

price_by_age_model = lm.LinearRegression()

price_by_age_model.fit(X=selected_model_df[["Age"]], y=selected_model_df[["Price"]])

age_range = [[selected_model_df["Age"].min()], [selected_model_df["Age"].max()]]

print(age_range)

predicted_price_by_age = price_by_age_model.predict(X=age_range)

plt.plot(age_range, predicted_price_by_age, color="r")

print(price_by_age_model.coef_)
print(price_by_age_model.intercept_)

plt.show()
