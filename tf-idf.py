import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Читаем файл
train = pd.read_csv("data/output/train.csv")
test = pd.read_csv("data/output/test.csv")

# Векторизуем
vectorizer = TfidfVectorizer()
X_train_vector = vectorizer.fit_transform(train["text"])
X_test_vector = vectorizer.transform(test["text"])

# Обучаем логистическую регрессию
model_lr = LogisticRegression(max_iter=1000, class_weight="balanced")
model_lr.fit(X_train_vector, train["label"])
y_pred = model_lr.predict(X_test_vector)

# Метрика качества
print("accuracy score:", accuracy_score(test["label"], y_pred))
print("precision score:", precision_score(test["label"], y_pred, average="weighted"))
print("recall score:", recall_score(test["label"], y_pred, average="weighted"))
print("F1 score:", f1_score(test["label"], y_pred, average="weighted"))

# Сохраним модели
with open("models/tf_idf_start.pkl", "wb") as file:
    pickle.dump(vectorizer, file=file)
with open("models/logreg_start.pkl", "wb") as file:
    pickle.dump(vectorizer, file=file)
