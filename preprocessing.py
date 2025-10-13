import logging

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logging.info("Делаю обработку данных")

df = pd.read_csv("data/processing/data_read.csv")
df = df.rename(columns={"Эмоциональная окраска": "label", "Комментарии": "text"})


df["label"] = df["label"].str.strip().str.lower()
df["text"] = df["text"].apply(lambda x: x.replace("\n", ""))

logging.info("Делаю визуализацию")

df["label"].value_counts().plot(kind="bar")
plt.xlabel = "Класс"
plt.ylabel = "Количество"
plt.title = "Распределение классов"
print(plt.show())

logging.info("Делаю трейн тест сплит")

X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    df["text"], df["label"], test_size=0.15, random_state=52, stratify=df["label"]
)

logging.info("Делаю encoder")

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train_raw)
y_test = encoder.transform(y_test_raw)

pd.concat([X_train, y_train], axis=1).to_csv("data/output/train.csv")
pd.concat([X_test, y_test], axis=1).to_csv("data/output/test.csv")
