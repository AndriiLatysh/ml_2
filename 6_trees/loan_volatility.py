import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as sk_trees


def convert_to_numeric_values(df):
    converted_df = df.copy()
    converted_df = converted_df.replace({"history": {"poor": 1, "fair": 2, "excellent": 3},
                                         "income": {"low": 1, "high": 2}})

    return converted_df


plt.figure(figsize=(18, 10))
loan_df = pd.read_csv("data/loans.csv")
print(loan_df)

numeric_loan_df = convert_to_numeric_values(loan_df)

features = ["history", "term", "income"]
X = numeric_loan_df[features]
y = numeric_loan_df[["risk"]]

loan_decision_tree = sk_trees.DecisionTreeClassifier()

loan_decision_tree.fit(X, y)

sk_trees.plot_tree(loan_decision_tree, feature_names=features, class_names=["high", "low"], filled=True, rounded=True)

plt.show()
