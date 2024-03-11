# IMDB Sentiment Classification
This project aims to classify IMDb movie reviews into positive or negative sentiments using Natural Language Processing (NLP) techniques. The sentiment classification is performed on a dataset of IMDb reviews, and various machine learning models are explored to achieve optimal performance.

## Overview
The project follows these main steps:
1. Data Preprocessing: The IMDb reviews undergo preprocessing steps including the removal of HTML tags, stopwords, non-alphanumeric characters, and lemmatization.
2. Feature Engineering: The preprocessed text data is transformed into numerical features using vectorization techniques such as CountVectorizer and TF-IDF.
3. Model Selection: Several machine learning models are tested on the preprocessed data, including Naive-Bayes, Random Forest, XGBoost, and Logistic Regression. Logistic Regression is identified as the top-performing model based on classification metrics.
4. Deep Learning Approach: A deep learning approach is implemented using a pre-trained BERT model from Hugging Face. The model is fine-tuned on the IMDb dataset for one epoch, demonstrating superior performance compared to traditional machine learning models.

## Technologies Used
* Pandas: Utilized for data manipulation and analysis.
* NLTK: Employed for text preprocessing tasks such as stopwords removal and lemmatization.
* Scikit-learn: Used for implementing machine learning models, feature engineering, and evaluation of classification metrics.
* Hugging Face: Leveraged for accessing pre-trained BERT model and fine-tuning it for sentiment classification.
* MLflow: Integrated for tracking model experiments and performance metrics.
* FastAPI: Utilized for building the RESTful API for model deployment.
* Docker: Deployed the model using Docker containers for efficient and scalable deployment.
