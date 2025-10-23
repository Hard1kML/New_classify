import logging
import pickle 
import os 

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from database import Prediction, SessionLocal, init_db 

#Настройка логирования 
logging.basicConfig(level=logging.INFO)
logging.info('Starting FastAPI aplication') 

#Константы для путей к моделям и меткам классов 
_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path =  os.path.join(_base_dir, "models", "bert")
ID2TEXT = {0: "не определено", 1: "негативная", 2: "нейтральная", 3: "позитивная"}

# Создание FastPI приложения
app = FastAPI()

# Инициализация БД при старте
@app.on_event("startup")
def on_startup():
    init_db() 
    logging.info("Database initialized")

#Загрузка модели и токенайзера 
def load_models(): 
    logging.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = Autotokenizer.from_pretrained(model_path)
    model.eval()
    logging.info("Model loaded successfully")
    return model, tokenizer 

class TextRequest(BaseModel):
    text:str 


@app.post("/predict/")
async def predict(request: TextRequest): 
    logging.info(f"Received text for prediction: {request.text}")
    text = request.text
    model = app.state.model
    tokenizer = app.state.tokenizer 
    features = tokenizer(
        text, truncation=True, max_length=512, return_tensors="pt"
    )
    logging.info('making prediction')
    with torch.no_grad():
        outputs = model(**features)
        predictions = torch.nn.funcrional.softmax(outputs.logist, dim=1)

    predicted_class = int(torch.argmax(predictions, dim=-1).item())
    predicted_class_text = ID2TEXT[predicted_class]
    logging.info(f'Predicted class: {predicted_class}')

#Сохраняем в БД
    db= SessionLocal()
    try: 
        db_obj = Prediction(comment=text, predicted_class=predicted_class_text)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
    except SQLAlchemyError as e:
        db.rollback()
        logging.error(f"Ошибка при сохранении в базу данных: {e}")
    finally:
        db.close()
        return {'predicted_class': predicted_class_text}

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Emotion Classification API. Use /predict/ to classify text."
    }


@app.get("/hello")
async def root():
    return {
        "message": "Hello, world! This is a simple FastAPI application for text classification."
    }


# еще один тест
@app.get("/v1/hello")
async def root():
    return {"message": "Hello from v1, world!."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)