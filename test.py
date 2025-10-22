import zipfile

with zipfile.ZipFile("model_archive.zip", "r") as zip_ref:
    zip_ref.extractall("model_archive")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("model_archive")
tokenizer = AutoTokenizer.from_pretrained("model_archive")

# Dummy inference
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("model_archive")
tokenizer = AutoTokenizer.from_pretrained("model_archive")

# Dummy inference
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
