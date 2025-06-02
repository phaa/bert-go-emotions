from fastapi import FastAPI
import torch
from transformers import BertForSequenceClassification, BertTokenizer

app = FastAPI()

model = BertForSequenceClassification.from_pretrained("../goemotions-bert")
tokenizer = BertTokenizer.from_pretrained("../goemotions-bert")
model.eval()

@app.post("/predict")
def predict(text_in: str):
    inputs = tokenizer(text_in.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(outputs.logits)
        preds = (probs > 0.5).int()
    return {"emotions": preds.tolist()}