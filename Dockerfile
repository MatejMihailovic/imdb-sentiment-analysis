# syntax=docker/dockerfile:1

FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords punkt wordnet averaged_perceptron_tagger
WORKDIR /app/matej_scripts
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]