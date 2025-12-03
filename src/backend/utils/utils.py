import logging
import pickle

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import LR_MODEL_PATH, MODEL_PATH, MODEL_PATH_POSIX, TFIDF_PATH


# Загрузка моделей
def load_models():
    logging.info("Loading models...")
    with open(LR_MODEL_PATH, "rb") as f:
        lr_model = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    logging.info("Models loaded successfully")
    return lr_model, tfidf


# Загрузка моделей NN
def load_models_NN():
    logging.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    logging.info("Model loaded successfully")
    return model, tokenizer
