import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from joblib import dump

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Dictionary of common abbreviations
abbreviation_dict = {
    "he's": "he is", "there's": "there is", "We're": "We are", "That's": "That is", 
    "won't": "will not", "they're": "they are", "Can't": "Cannot", "wasn't": "was not", 
    "don\x89Ûªt": "do not", "aren't": "are not", "isn't": "is not", "What's": "What is", 
    "haven't": "have not", "hasn't": "has not", "There's": "There is", "He's": "He is", 
    "It's": "It is", "You're": "You are", "I'M": "I am", "shouldn't": "should not", 
    "wouldn't": "would not", "i'm": "I am", "I\x89Ûªm": "I am", "I'm": "I am", 
    "Isn't": "is not", "Here's": "Here is", "you've": "you have", "you\x89Ûªve": "you have", 
    "we're": "we are", "what's": "what is", "couldn't": "could not", "we've": "we have", 
    "it\x89Ûªs": "it is", "doesn\x89Ûªt": "does not", "It\x89Ûªs": "It is", 
    "Here\x89Ûªs": "Here is", "who's": "who is", "I\x89Ûªve": "I have", "y'all": "you all", 
    "can\x89Ûªt": "cannot", "would've": "would have", "it'll": "it will", 
    "we'll": "we will", "wouldn\x89Ûªt": "would not", "We've": "We have", "he'll": "he will", 
    "Y'all": "You all", "Weren't": "Were not", "Didn't": "Did not", "they'll": "they will", 
    "they'd": "they would", "DON'T": "DO NOT", "That\x89Ûªs": "That is", "they've": "they have", 
    "i'd": "I would", "should've": "should have", "You\x89Ûªre": "You are", "where's": "where is", 
    "Don\x89Ûªt": "Do not", "we'd": "we would", "i'll": "I will", "weren't": "were not", 
    "They're": "They are", "Can\x89Ûªt": "Cannot", "you\x89Ûªll": "you will", "I\x89Ûªd": "I would", 
    "let's": "let us", "it's": "it is", "can't": "cannot", "don't": "do not", "you're": "you are", 
    "i've": "I have", "that's": "that is", "i'll": "I will", "doesn't": "does not", "i'd": "I would", 
    "didn't": "did not", "ain't": "am not", "you'll": "you will", "I've": "I have", "Don't": "do not", 
    "I'll": "I will", "I'd": "I would", "Let's": "Let us", "you'd": "You would", "It's": "It is", 
    "Ain't": "am not", "Haven't": "Have not", "Could've": "Could have", "youve": "you have", 
    "donå«t": "do not"
}

def remove_abbreviations(data):
    # Replace abbreviations with their full form
    for abb, full in abbreviation_dict.items():
        data = re.sub(r"\b{}\b".format(abb), full, data)
    
    return data 

def remove_punctuations(data):
    # Remove punctuations
    punct_tag = re.compile(r'[^\w\s]')
    data = punct_tag.sub(r'', data)
    return data

def remove_html(data):
    # Remove HTML syntaxes
    return data.replace('<br /><br />', ' ')

def remove_url(data):
    # Remove URL data
    url_clean = re.compile(r"https://\S+|www\.\S+")
    data = url_clean.sub(r'', data)
    return data

def remove_emojis(data):
    # Remove Emojis
    emoji_clean = re.compile("["
                            u"\U0001F600-\U0001F64F"  
                            u"\U0001F300-\U0001F5FF"  
                            u"\U0001F680-\U0001F6FF"  
                            u"\U0001F1E0-\U0001F1FF"  
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
    data = emoji_clean.sub(r'', data)
    url_clean = re.compile(r"https://\S+|www\.\S+")
    data = url_clean.sub(r'', data)
    return data

def remove_non_alphanumeric(data):
    # Remove non-alphanumeric characters
    return re.sub(r'[^a-zA-Z0-9\s]', '', data)

def remove_stopwords(text):
    # Remove stopwords
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def remove_numbers(text):
    # Remove numbers
    return re.sub(r'\d+', '', text)

def lemmatize_tokens(doc):
    # Function to lemmatize a list of tokens
    return [token.lemma_ for token in nlp(doc)]


def preprocess_doc(text):
    text = remove_abbreviations(text)
    text = remove_non_alphanumeric(text)
    text = remove_emojis(text)
    text = remove_punctuations(text)
    text = text.lower()
    text = remove_stopwords(text)
    text = remove_numbers(text)
    text = lemmatize_tokens(text)

    return ' '.join(text)

def tfidf_transform(text_data, max_features=20000, min_df=0.0, max_df=1.0, ngram_range=(1,2), use_idf=True):
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df, ngram_range=ngram_range, use_idf=use_idf, lowercase=False)
    vectorizer_filename = './models/basic_models/tfidf_vectorizer.joblib'
    tfidf_data = tfidf.fit_transform(text_data)
    dump(tfidf, vectorizer_filename)

    return tfidf_data 

def count_vect_transform(text_data, max_features=20000, min_df=0.0, max_df=1.0, ngram_range=(1,2)):
    # Count Vectorization
    count_vect = CountVectorizer(max_features=max_features, min_df=min_df, max_df=max_df, ngram_range=ngram_range, lowercase=False)
    vectorizer_filename = './models/basic_models/count_vectorizer.joblib'
    count_vect_data = count_vect.fit_transform(text_data)
    dump(count_vect, vectorizer_filename)
    return count_vect_data