import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logging.info("Cчитываю файлы")


urls = {
    "UX": "https://docs.google.com/spreadsheets/d/1P9EfZ3nWMFqp7q04alhIatXZRWWodirq/export?format=xlsx&gid=1642103472",
    "GP": "https://docs.google.com/spreadsheets/d/1P9EfZ3nWMFqp7q04alhIatXZRWWodirq/export?format=xlsx&gid=1024446129",
    "AS": "https://docs.google.com/spreadsheets/d/1P9EfZ3nWMFqp7q04alhIatXZRWWodirq/export?format=xlsx&gid=923763835",
}

dfs = []
for name, url in urls.items():
    df = pd.read_excel(url, engine="openpyxl")
    if {"Комментарии", "Эмоциональная окраска"}.issubset(df.columns):
        dfs.append(df[["Комментарии", "Эмоциональная окраска"]])
        logging.info(f"Лист {name} загружен")
    else:
        logging.warning(f"В листе {name} отсутствуют нужные колонки")

df = pd.concat(dfs, ignore_index=True)


logging.info("Готово")
df.to_csv("./data/processing/data_read.csv", index=False)
print(df.shape)
