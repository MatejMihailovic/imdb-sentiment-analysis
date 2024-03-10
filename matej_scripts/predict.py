from fastapi import FastAPI, HTTPException
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import os 
from joblib import load
import text_preprocessing as tp
from collections import Counter
import uvicorn

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
# Change the working directory to the root folder
os.chdir(root_dir)

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

path = './models/bert-base-cased'
bert_model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)

lr = load('./models/basic_models/logistic_regression_model_tfidf.joblib')
tfidf = load('./models/basic_models/tfidf_vectorizer.joblib')

@app.post("/bert_predict_sentiment/")
async def predict_sentiment_bert(sentence: str):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = bert_model.forward(**inputs, output_hidden_states=True)

    # Get probabilities using softmax
    probabilities = F.softmax(outputs.logits, dim=1)

    # Check if the probability at index 0 is higher
    if probabilities[0][0] > probabilities[0][1]:
        sentiment = "Negative"
    else:
        sentiment = "Positive"

    return {"sentence": sentence, "sentiment": sentiment}

@app.post("/lr_predict_sentiment/")
async def predict_sentiment_lr(sentence: str):
    sentence_cleaned = tp.preprocess_doc(sentence)

    sentence_tfidf = tfidf.transform([sentence_cleaned])

    prediction = lr.predict(sentence_tfidf)

    predicted_sentiment = "positive" if prediction == 1 else "negative"

    return {"sentence": sentence, "predicted_sentiment": predicted_sentiment}

