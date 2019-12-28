import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_ms
import sklearn.preprocessing as sk_preprocessing
import sklearn.neural_network as sk_nn
import sklearn.svm as sk_svm
import sklearn.ensemble as sk_ensemble


def plot_model(model, title, scaler, qualifies_double_grade, subplot):
    print("Starting plotting {}".format(title))

    subplot.set_title(title)

    qualified_candidates = qualifies_double_grade[qualifies_double_grade["qualifies"] == 1]
    unqualified_candidates = qualifies_double_grade[qualifies_double_grade["qualifies"] == 0]

    max_grade = 101
    prediction_points = []

    for technical_grade in range(max_grade):
        for english_grade in range(max_grade):
            prediction_points.append(scaler.transform([[technical_grade, english_grade]]))
    prediction_points = np.array(prediction_points).reshape(-1, 2)

    probability_levels = model.predict_proba(prediction_points)[:, 1]
    probability_levels = np.array(probability_levels).reshape(101, 101)

    subplot.contourf(probability_levels, cmap="rainbow")

    subplot.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="w")
    subplot.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="k")

    print("Finished plotting {}".format(title))


fig, subplots = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

qualifies_by_double_grade = pd.read_csv("data/double_grade_reevaluated.csv")

X = qualifies_by_double_grade[["technical_grade", "english_grade"]]
y = qualifies_by_double_grade["qualifies"]

X_scaler = sk_preprocessing.StandardScaler()
# X_scaler = sk_preprocessing.MinMaxScaler()
X = X_scaler.fit_transform(X)

k_folds = sk_ms.KFold(n_splits=4, shuffle=True)

ann_model = sk_nn.MLPClassifier(hidden_layer_sizes=(10, 10), activation="tanh", max_iter=100000)
ann_results = sk_ms.cross_val_score(ann_model, X, y, cv=k_folds)
print("Neural Network accuracy: {:.2f} %".format(ann_results.mean() * 100))

svm_model = sk_svm.SVC(kernel="rbf", gamma="scale", probability=True)
svm_results = sk_ms.cross_val_score(svm_model, X, y, cv=k_folds)
print("Support Vector Machine accuracy: {:.2f} %".format(svm_results.mean() * 100))

rfc_model = sk_ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=20)
rfc_results = sk_ms.cross_val_score(rfc_model, X, y, cv=k_folds)
print("Random Forest accuracy: {:.2f} %".format(rfc_results.mean() * 100))

estimators = []

estimators.append(("ANN", ann_model))
estimators.append(("SVM", svm_model))
estimators.append(("RF", rfc_model))

ensemble_model = sk_ensemble.VotingClassifier(estimators, voting="soft")
ensemble_results = sk_ms.cross_val_score(ensemble_model, X, y, cv=k_folds)
print()
print("Voting Classifier accuracy: {:.2f} %".format(ensemble_results.mean() * 100))
print()

ann_model.fit(X, y)
svm_model.fit(X, y)
rfc_model.fit(X, y)
ensemble_model.fit(X, y)

plot_model(ann_model, "Neural Network", X_scaler, qualifies_by_double_grade, subplots[0][0])
plot_model(svm_model, "Support Vector Machine", X_scaler, qualifies_by_double_grade, subplots[0][1])
plot_model(rfc_model, "Random Forest", X_scaler, qualifies_by_double_grade, subplots[0][2])
plot_model(ensemble_model, "Voting Classifier", X_scaler, qualifies_by_double_grade, subplots[1][1])

plt.show()
