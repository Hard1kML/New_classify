import logging
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

train = pd.read_csv("data/output/train.csv")
test = pd.read_csv("data/output/test.csv")

pipeline = Pipeline(
    [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))]
)

param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l2"],
    "clf__solver": ["saga", "lbfgs"],
}

logging.info("start gridsearchCV")
Grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1)
Grid_search.fit(train["text"], train["label"])

logging.info("Лучшие параметры:", Grid_search.best_params_)

y_pred = Grid_search.predict(test["text"])

print("Accuracy score:", accuracy_score(test["label"], y_pred))
print("Precision score:", precision_score(test["label"], y_pred, average="weighted"))
print("Recall score:", recall_score(test["label"], y_pred, average="weighted"))
print("F1 score:", f1_score(test["label"], y_pred, average="weighted"))

with open("models/tf_idf_adv.pkl", "wb") as file:
    pickle.dump(Grid_search.best_estimator_.named_steps["tfidf"], file)
with open("models/lr_adv.pkl", "wb") as file:
    pickle.dump(Grid_search.best_estimator_.named_steps["clf"], file)
