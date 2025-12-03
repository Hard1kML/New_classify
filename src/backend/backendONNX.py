import logging
import sys, os
from pathlib import Path
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from transformers import AutoTokenizer
import onnxruntime as ort
from src.backend.database import Prediction, SessionLocal, init_db



# Настройка логирования

logging.basicConfig(level=logging.INFO)
logging.info("Starting FastAPI application")

# Пути и константы

from pathlib import Path

# Папка backend
BASE_DIR = Path(__file__).resolve().parent

# Папка models внутри backend
MODEL_PATH = BASE_DIR / "models" / "bert"

# Путь к ONNX файлу
ONNX_MODEL_PATH = MODEL_PATH / "model.onnx"
ID2TEXT = {0: "не определено", 1: "негативная", 2: "нейтральная", 3: "позитивная"}


# Создание FastAPI приложения

app = FastAPI()


# Инициализация базы данных и модели

@app.on_event("startup")
def on_startup():
    try:
        init_db()
        logging.info("Database initialized")

        # Токенизатор
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # ONNX Runtime сессия
        ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH))

        # Сохраняем в app.state
        app.state.tokenizer = tokenizer
        app.state.ort_session = ort_session

        logging.info("ONNX model and tokenizer attached to app state")
    except Exception as e:
        logging.exception(f"Ошибка инициализации при старте: {e}")


# Схема данных запроса

class TextRequest(BaseModel): 
    text: str


# Endpoint для предсказания через ONNX

@app.post("/predict/")
async def predict(request: TextRequest):
    logging.info(f"Received text for prediction: {request.text}")
    text = request.text.strip()
    if not text:
        return {"error": "Пустой ввод"}

    tokenizer = app.state.tokenizer
    ort_session = app.state.ort_session

    # Токенизация
    features = tokenizer(
        text,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="np"  # ONNX Runtime работает с numpy
    )

    # Приведение к int64
    inputs_onnx = {
        "input_ids": features["input_ids"].astype("int64"),
        "attention_mask": features["attention_mask"].astype("int64")
    }

    # Предсказание через ONNX
    logits = ort_session.run(None, inputs_onnx)[0]
    predictions = torch.nn.functional.softmax(torch.tensor(logits), dim=1)
    predicted_class = int(torch.argmax(predictions, dim=-1).item())
    predicted_class_text = ID2TEXT.get(predicted_class, "не определено")

    logging.info(f"Predicted class: {predicted_class_text}")

    # Сохраняем в БД
    db = SessionLocal()
    try:
        db_obj = Prediction(comment=text, predicted_class=predicted_class_text)
        db.add(db_obj)
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        logging.error(f"Ошибка при сохранении в базу данных: {e}")
    finally:
        db.close()

    return {"predicted_class": predicted_class_text, "logits": predictions.tolist()}

# Тестовые маршруты

@app.get("/")
async def root():
    return {"message": "Welcome to the Emotion Classification API. Use /predict/ to classify text."}

# Точка входа

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)