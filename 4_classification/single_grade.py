import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.metrics as sm


qualifies_single_grade = pd.read_csv("data/single_grade.csv")
qualifies_single_grade.sort_values(by="grade", inplace=True)
# print(qualifies_single_grade)

qualified_candidates = qualifies_single_grade[qualifies_single_grade["qualifies"] == 1]
unqualified_candidates = qualifies_single_grade[qualifies_single_grade["qualifies"] == 0]

plt.scatter(qualified_candidates["grade"], qualified_candidates["qualifies"], color="g")
plt.scatter(unqualified_candidates["grade"], unqualified_candidates["qualifies"], color="r")

X = qualifies_single_grade[["grade"]]
y = qualifies_single_grade["qualifies"]

qualification_model = lm.LogisticRegression(solver="lbfgs")
qualification_model.fit(X, y)

modeled_qualification = qualification_model.predict(X)
modeled_qualification_probability = qualification_model.predict_proba(X)[:, 1]

qualifies_single_grade["probability"] = modeled_qualification_probability

plt.plot(X, modeled_qualification, color="b")
plt.plot(X, modeled_qualification_probability, color="c")

print(qualifies_single_grade)

confusion_matrix = sm.confusion_matrix(y, modeled_qualification)

print(confusion_matrix)

plt.show()
