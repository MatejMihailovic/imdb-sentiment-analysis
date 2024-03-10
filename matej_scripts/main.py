import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from model_training_evaluation import Model
import text_preprocessing as tp
import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
# Change the working directory to the root folder
os.chdir(root_dir)

def grid_search_for_best_params(model, param_grid, X_train, y_train, cv=3):
    """
    Perform grid search to find the best hyperparameters for the given model.
    
    Args:
        model: The model object to tune.
        param_grid (dict): The parameter grid to search over.
        X_train: The training data features.
        y_train: The training data labels.
        cv (int): Number of cross-validation folds.

    Returns:
        dict: Best hyperparameters found by grid search.
    """
    grid_search = GridSearchCV(model, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    """
    Train and evaluate the given model on both CountVectorizer and TF-IDF transformed data.

    Args:
        model_name (str): Name of the model.
        model: The model object to evaluate.
        X_train: Training data features.
        y_train: Training data labels.
        X_test: Test data features.
        y_test: Test data labels.
    """
    if "Logistic_Regression" in model_name:
        label_encoder = LabelEncoder()

        # Define parameter grid for logistic regression
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 
                      'penalty': ['l1', 'l2'], 
                      'solver': ['liblinear', 'saga']}

        # Find best hyperparameters for logistic regression
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)
        best_params = grid_search_for_best_params(LogisticRegression(), param_grid, X_train, y_train, 5)

        # Create models with best hyperparameters
        model = Model(model_name, LogisticRegression(**best_params))

        with open(f'./models/basic_models/best_params_{model_name}.json', 'w') as file:
            json.dump(best_params, file)
    elif "XGBClassifier" in model_name:
        label_encoder = LabelEncoder()

        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)

        # Define parameter grid for XGBoost
        param_grid = {'max_depth': [3, 4, 5], 
                      'learning_rate': [0.1, 0.01, 0.05], 
                      'n_estimators': [100, 200, 300, 400, 500], 
                      'gamma': [0, 0.1, 0.2]}

        # Find best hyperparameters for XGBoost
        best_params = grid_search_for_best_params(XGBClassifier(tree_method='hist', device='cuda'), param_grid, X_train, y_train)

        with open(f'./models/basic_models/best_params_{model_name}.json', 'w') as file:
            json.dump(best_params, file)

        # Create models with best hyperparameters
        model= Model(model_name, XGBClassifier(**best_params))
    elif "Random_Forest_Classifier" in model_name:
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        # Find best hyperparameters for Random Forest
        best_params = grid_search_for_best_params(RandomForestClassifier(), param_grid, X_train, y_train)

        with open(f'./models/basic_models/best_params_{model_name}.json', 'w') as file:
            json.dump(best_params, file)
        # Create models with best hyperparameters
        model = Model(model_name, RandomForestClassifier(**best_params))
    else:
        # Create models with default hyperparameters
        model = Model(model_name, model)

    # Evaluate models
    model.evaluate_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    #data = IMDbDataset('./data/imdb_dataset.csv')
    #data.to_csv('./data/imdb_cleaned.csv', index=False)
    # data.display_word_cloud(data['review'], './images/plots/all_reviews_word_cloud.jpg')
    # data.good_bad_reviews()

    data = pd.read_csv('./data/imdb_cleaned.csv')
    
    X_cv = tp.count_vect_transform(data['clean_review'], max_features=20000, ngram_range=(1,2), min_df=0.2, max_df=0.8)
    X_tf_idf = tp.tfidf_transform(data['clean_review'], max_features=50000, ngram_range=(1,2), min_df=0.0, max_df=1.0, use_idf=True)
    y = data['sentiment']

    # Train-test split
    X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_cv, y, test_size=0.2, random_state=42)
    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tf_idf, y, test_size=0.2, random_state=42)

    # Evaluation of basic ml models
    evaluate_model("Naive-Bayes_(CV)", MultinomialNB(), X_train_cv.toarray(), y_train_cv, X_test_cv.toarray(), y_test_cv)
    evaluate_model("Naive-Bayes_(TF-IDF)", MultinomialNB(), X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf)

    evaluate_model("Random_Forest_Classifier_(CV)", RandomForestClassifier(), X_train_cv, y_train_cv, X_test_cv, y_test_cv)
    evaluate_model("Random_Forest_Classifier_(TF-IDF)", RandomForestClassifier(), X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf)

    evaluate_model("Logistic_Regression_(CV)", LogisticRegression(), X_train_cv, y_train_cv, X_test_cv, y_test_cv)
    evaluate_model("Logistic_Regression_(TF-IDF)", LogisticRegression(), X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf)

    evaluate_model("XGBClassifier_(CV)", XGBClassifier(tree_method='hist', device='cuda'), X_train_cv, y_train_cv, X_test_cv, y_test_cv)
    evaluate_model("XGBClassifier_(TF-IDF)", XGBClassifier(tree_method='hist', device='cuda'), X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf)
    
    
