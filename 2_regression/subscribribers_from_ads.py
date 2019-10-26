import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm


subs_from_ads_df = pd.read_csv("data/subscribers_from_ads.csv")
plt.scatter(subs_from_ads_df[["promotion_budget"]], subs_from_ads_df[["subscribers"]])

subs_from_ads_model = lm.LinearRegression()
subs_from_ads_model.fit(X=subs_from_ads_df[["promotion_budget"]], y=subs_from_ads_df[["subscribers"]])
modeled_values = subs_from_ads_model.predict(X=subs_from_ads_df[["promotion_budget"]])

plt.plot(subs_from_ads_df[["promotion_budget"]], modeled_values)

print(subs_from_ads_model.coef_)
print(subs_from_ads_model.intercept_)

plt.show()
