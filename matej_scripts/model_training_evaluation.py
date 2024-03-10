import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from utils import mlflow_create_experiment
import matplotlib.pyplot as plt
from joblib import dump

class Model:
    def __init__(self, model_name, model):
        self.model_name = model_name
        self.model = model

    def get_params(self):
        return self.model.get_params()
    
    def get_name(self):
        return self.model_name
    
    def compute_metrics(self, y_true, y_pred):
        # Compute classification report and confusion matrix
        cr = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        return cr, cm

    def log_metrics(self, class_metrics, confusion_matrix):
        accuracy = class_metrics.pop("accuracy")
        tn, fp, fn, tp = confusion_matrix.ravel()

        # Log confusion matrix
        mlflow.log_metric("True Positives", tp)
        mlflow.log_metric("True Negatives", tn)
        mlflow.log_metric("False Positives", fp)
        mlflow.log_metric("False Negatives", fn)

        # Log accuracy
        mlflow.log_metric("Accuracy", accuracy)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['Negative', 'Positive'])
        disp.plot()
        plt.savefig(f"./images/plots/{self.model_name}_confusion_matrix.png")
        plt.close()

        # Log class metrics
        for class_or_avg, metrics_dict in class_metrics.items():
            for metric, value in metrics_dict.items():
                if "support" not in metric:
                    if class_or_avg == "0":
                        mlflow.log_metric('negative_' + metric, value)
                    elif class_or_avg == "1":
                        mlflow.log_metric('positive_' + metric, value)
                    else:
                        mlflow.log_metric(class_or_avg + '_' + metric, value)

    def evaluate_model(self, X_train, y_train, X_test, y_test):
        tracking_uri = "http://192.168.66.221:20002/"
        experiment_name = "matej_praksa_sa"

        mlflow.set_tracking_uri(tracking_uri)

        experiment_id = mlflow_create_experiment(experiment_name=experiment_name)
        with mlflow.start_run(experiment_id=experiment_id, run_name=self.model_name):
            # Train the model
            self.model.fit(X_train, y_train)
            
            model_filename = f'./models/basic_models/{self.model_name}.joblib'
            dump(self.model, model_filename)

            # Evaluate the model
            y_pred = self.model.predict(X_test)
            
            # Compute evaluation metrics
            class_metrics, confusion_matrix = self.compute_metrics(y_test, y_pred)
            
            # Log evaluation metrics
            self.log_metrics(class_metrics, confusion_matrix)
            
            # Log model parameters
            mlflow.log_params(self.get_params())
            
            # Log the trained model
            mlflow.sklearn.log_model(self.model, self.get_name())

            # End the MLflow run
            mlflow.end_run()

    
