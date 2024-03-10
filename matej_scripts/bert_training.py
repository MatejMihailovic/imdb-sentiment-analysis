import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import torch
import evaluate
import os

pd.set_option('display.max_colwidth', None)

import warnings
warnings.filterwarnings("ignore")

os.environ['DISABLE_MLFLOW_INTEGRATION'] = "TRUE"

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
# Change the working directory to the root folder
os.chdir(root_dir)

class IMDbDataset:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.data = self.data.rename(columns={'sentiment': 'label'})
        self.data['label'] = self.data['label'].map(lambda x: 1 if x == 'positive' else 0)

        # Removes HTML syntaxes
        self.data['review'] = self.data['review'].apply(lambda x: x.replace('<br /><br />', ' '))

    def train_eval_test_split(self, test_size=0.2, random_state=42):
        train_data, test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)
        train_data, eval_data = train_test_split(train_data, test_size=test_size, random_state=random_state)
        return train_data, eval_data, test_data


class IMDbModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.metric = evaluate.load('accuracy')

    def tokenize_data(self, data):
        return self.tokenizer(data['review'],
                              padding='max_length',
                              truncation=True,
                              return_tensors='pt')

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    experiment_name = "matej_praksa_bert5"
    run_name = "bert_model"

    #Change if needed
    mlflow_uri = "http://192.168.66.221:20002/"
    
    mlflow.set_tracking_uri(mlflow_uri)
    dataset = IMDbDataset('./data/imdb_dataset.csv')
    train_data, eval_data, test_data = dataset.train_eval_test_split()

    model = IMDbModel("bert-base-cased")

    train_dataset = Dataset.from_pandas(train_data)
    eval_dataset = Dataset.from_pandas(eval_data)
    test_dataset = Dataset.from_pandas(test_data)

    train_dataset = train_dataset.map(model.tokenize_data, batched=True)
    eval_dataset = eval_dataset.map(model.tokenize_data, batched=True)
    test_dataset = test_dataset.map(model.tokenize_data, batched=True)

    training_args = TrainingArguments(output_dir='./models/checkpoints/bert-base-cased-sentiment',
                                      evaluation_strategy='steps',
                                      per_device_train_batch_size=8,
                                      per_device_eval_batch_size=8,
                                      num_train_epochs=3,
                                      save_strategy='steps',
                                      save_steps=500,
                                      logging_steps=500)

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='bert_model'):
        trainer.train()

        trainer.save_model('./models/bert-base-cased')


        mlflow.log_param('model_name', model.model_name)
        mlflow.log_param('max_length', 512)
        mlflow.log_param('batch_size', 8)
        mlflow.log_param('num_train_samples', len(train_dataset))
        mlflow.log_param('num_eval_samples', len(eval_dataset))
        mlflow.log_param('num_test_samples', len(test_dataset))

        # Log eval metrics
        eval_metrics = trainer.evaluate()
        for key, value in eval_metrics.items():
            mlflow.log_metric(key, value)

        # Generate classification report
        test_predictions = trainer.predict(test_dataset)
        test_logits, test_labels = test_predictions.predictions, test_predictions.label_ids
        test_logits = torch.argmax(torch.tensor(test_logits), dim=1).tolist()
        test_labels = list(test_labels)

        cr = classification_report(test_labels, test_logits, output_dict=True)

        cm = confusion_matrix(test_labels, test_logits)

        tn, fp, fn, tp = cm.ravel()

        mlflow.log_metric("True Positives", tp)
        mlflow.log_metric("True Negatives", tn)
        mlflow.log_metric("False Positives", fp)
        mlflow.log_metric("False Negatives", fn)

        # Log model metrics
        mlflow.log_metric("Accuracy", cr.pop("accuracy"))
        for class_or_avg, metrics_dict in cr.items():
            for metric, value in metrics_dict.items():
                if "support" not in metric:
                    if class_or_avg == '0':
                        mlflow.log_metric(metric + '_negative', value)
                    elif class_or_avg == '1':
                        mlflow.log_metric(metric + '_positive', value)
                    else:
                        mlflow.log_metric(metric + '_' + class_or_avg, value)
                    

        print("Classification Report:")
        print(classification_report(test_labels, test_logits))

        test_dataset_with_predictions = pd.DataFrame(test_dataset).copy()
        test_dataset_with_predictions["predicted_labels"] = test_logits

        # Log the test dataset with appended predictions as an artifact
        test_dataset_with_predictions.to_csv("test_dataset_with_predictions.csv", index=False)
        mlflow.log_artifact("test_dataset_with_predictions.csv", artifact_path="test_data")

        mlflow.pytorch.log_model(model.model, artifact_path='huggingface_model', registered_model_name='huggingface_model')

        mlflow.end_run()
