import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = r"C:\Users\ser9e\New_classify\src\backend\models\bert"
ONNX_MODEL_PATH = r"C:\Users\ser9e\New_classify\src\backend\models\bert\model.onnx"

# Загружаем модель
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Пример входа
text = "Привет, как дела?"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")

# Экспорт в ONNX
torch.onnx.export(
    model, 
    (inputs["input_ids"], inputs["attention_mask"]),  # входы модели
    ONNX_MODEL_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size"}
    },
    opset_version=13,
)
print(f"ONNX модель сохранена в {ONNX_MODEL_PATH}")