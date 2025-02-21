import logging
import yaml
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
import pandas as pd
from datasets import Dataset
import numpy as np
import evaluate

# Load configuration
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NameDataset:
    def __init__(self, config):
        self.data_path = config['data_path']
        self.dataset = None
        self.tokenized_dataset = None
        self.tokenizer = None
        logger.info("Initialized NameDataset with data path: %s", self.data_path)

    def load(self):
        try:
            UKNameDataset = pd.read_csv(self.data_path, names=['first', 'last', 'gender', 'country'], header=None)
            self.dataset = Dataset.from_pandas(UKNameDataset)
            logger.info("Loaded dataset with %d records", len(UKNameDataset))
        except Exception as e:
            logger.error("Failed to load dataset: %s", e)
            raise

    def preprocess(self, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = self.dataset.class_encode_column("gender")
        self.dataset = self.dataset.train_test_split(test_size=0.2, stratify_by_column="gender")

        def preprocess_function(examples):
            return tokenizer(examples["first"], truncation=True)

        self.tokenized_dataset = self.dataset.map(preprocess_function, batched=True)
        self.tokenized_dataset = self.tokenized_dataset.rename_column("gender", "label")
        self.tokenized_dataset = self.tokenized_dataset.rename_column("first", "text")
        logger.info("Preprocessed dataset")

class NameGenderModel:
    def __init__(self, config):
        self.model_name = config['model_name']
        self.output_dir = config['output_dir']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.id2label = {0: "Female", 1: "Male"}
        self.label2id = {"Female": 0, "Male": 1}
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.trainer = None
        self.recall = evaluate.load("recall")
        self.accuracy = evaluate.load("accuracy")
        self.precision = evaluate.load("precision")
        logger.info("Initialized NameGenderModel with model name: %s", self.model_name)

    def train(self, tokenized_dataset):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=2e-5,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            num_train_epochs=10,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['test'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

        self.trainer.train()
        self.trainer.evaluate()
        logger.info("Training completed")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        pre = self.precision.compute(predictions=predictions, references=labels)["precision"]
        rec = self.recall.compute(predictions=predictions, references=labels)["recall"]
        acc = self.accuracy.compute(predictions=predictions, references=labels)
        return {"precision": pre, "recall": rec, "acc": acc}

    def save_model(self, path="./my_model"):
        self.trainer.save_model(path)
        logger.info("Model saved to %s", path)

    def classify(self, text):
        classifier = pipeline("text-classification", model="./my_model/")
        return classifier(text)

# Example usage:
config = load_config()
dataset = NameDataset(config)
dataset.load()
dataset.preprocess(AutoTokenizer.from_pretrained(config['model_name']))

model = NameGenderModel(config)
model.train(dataset.tokenized_dataset)
model.save_model()
print(model.classify('John Smith'))
