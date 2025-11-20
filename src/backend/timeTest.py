import time
import torch
import onnxruntime as ort
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import numpy as np

MODEL_PATH = Path(r"C:\Users\ser9e\New_classify\src\backend\models\bert")
TORCHSCRIPT_MODEL_PATH = Path(r"C:\Users\ser9e\New_classify\src\backend\models\bert\bert_ts.pt")
ONNX_MODEL_PATH = MODEL_PATH / "model.onnx"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


texts = ["Привет, как дела?"] * 50  # 50 одинаковых для стабильного среднего времени

# HF модель

hf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
hf_model.eval()

# TorchScript модель

ts_model = torch.jit.load(TORCHSCRIPT_MODEL_PATH)
ts_model.eval()

# ONNX модель

onnx_session = ort.InferenceSession(str(ONNX_MODEL_PATH))

# Функции предсказания

def predict_hf(texts):
    start = time.time()
    for text in texts:
        features = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        with torch.no_grad():
            logits = hf_model(**features).logits
            _ = torch.nn.functional.softmax(logits, dim=1)
    return time.time() - start

def predict_ts(texts):
    start = time.time()
    for text in texts:
        features = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        with torch.no_grad():
            logits, = ts_model(features["input_ids"], features["attention_mask"])
            _ = torch.nn.functional.softmax(logits, dim=1)
    return time.time() - start

def predict_onnx(texts):
    start = time.time()
    for text in texts:
        features = tokenizer(text, return_tensors="np", truncation=True, max_length=512, padding="max_length")
        inputs_onnx = {
            "input_ids": features["input_ids"].astype('int64'),
            "attention_mask": features["attention_mask"].astype('int64')
        }
        
        logits = onnx_session.run(None, inputs_onnx)[0]
        _ = torch.nn.functional.softmax(torch.tensor(logits), dim=1)
    return time.time() - start

# =========================
# Бенчмарк
# =========================
hf_time = predict_hf(texts)
ts_time = predict_ts(texts)
onnx_time = predict_onnx(texts)

print(f"HF model inference time (50 runs): {hf_time:.4f} sec, avg: {hf_time/50:.4f} sec/run")
print(f"TorchScript inference time (50 runs): {ts_time:.4f} sec, avg: {ts_time/50:.4f} sec/run")
print(f"ONNX inference time (50 runs): {onnx_time:.4f} sec, avg: {onnx_time/50:.4f} sec/run")