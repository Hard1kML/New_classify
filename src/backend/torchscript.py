import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "src/backend/models/bert"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Возвращаем только logits в виде кортежа
        return (outputs.logits,)


wrapped_model = WrappedModel(model)


logging.basicConfig(level=logging.INFO)
logging.info("loading model")

model.eval()

example_input_ids = torch.randint(0, tokenizer.vocab_size, (8, 512))
example_attention_mask = torch.ones_like(example_input_ids)

logging.info("tracing TorchScript")
traced_model = torch.jit.trace(
    wrapped_model, (example_input_ids, example_attention_mask)
)

OUT_PATH = f"{MODEL_PATH}/bert_ts.pt"

traced_model.save(OUT_PATH)

logging.info("TorchScript model saved in: {OUT_PATH}")
