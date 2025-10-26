import logging
import sys, os
from pathlib import Path
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from database import Prediction, SessionLocal, init_db 
from utils import load_models, load_models_NN


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logging.info("Starting FastAPI application")

# Константы для путей к моделям и меткам классов
MODEL_PATH = Path(r"C:\Users\ser9e\New_classify\src\backend\models\bert")
MODEL_PATH_POSIX = MODEL_PATH.resolve().as_posix()  # корректный путь для Transformers
ID2TEXT = {0: "не определено", 1: "негативная", 2: "нейтральная", 3: "позитивная"}

# Создание FastAPI приложения
app = FastAPI()

# =========================
# Инициализация базы данных и модели
# =========================
@app.on_event("startup")
def on_startup():
    try:
        init_db()
        logging.info("Database initialized")
        model, tokenizer = load_models_NN()
        app.state.model = model
        app.state.tokenizer = tokenizer
        logging.info("Model and tokenizer attached to app state")
    except Exception as e:
        logging.exception(f"Ошибка инициализации при старте: {e}")  # Логируем, но даем приложению стартовать для диагностики   






# =========================
# Схема данных запроса
# =========================
class TextRequest(BaseModel):
    text: str


# =========================
# Маршрут предсказания
# =========================
@app.post("/predict/")
async def predict(request: TextRequest):
    logging.info(f"Received text for prediction: {request.text}")
    text = request.text.strip()
    if not text:
        return {"error": "Пустой ввод"}

    model = app.state.model
    tokenizer = app.state.tokenizer

    features = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    logging.info("Making prediction...")
    with torch.no_grad():
        outputs = model(**features)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

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

    return {"predicted_class": predicted_class_text}


# =========================
# Тестовые маршруты
# =========================
@app.get("/")
async def root():
    return {"message": "Welcome to the Emotion Classification API. Use /predict/ to classify text."}


@app.get("/hello")
async def hello():
    return {"message": "Hello, world! This is a simple FastAPI application for text classification."}


@app.get("/v1/hello")
async def hello_v1():
    return {"message": "Hello from v1, world!."}


# =========================
# Точка входа
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
