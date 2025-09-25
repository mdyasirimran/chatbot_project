import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, Features, Value
import torch
import os

# Download NLTK resources (run once)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:  # <--- Correct
    print("NLTK 'wordnet' not found, attempting download...")
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("NLTK data download complete.")

lemmatizer = WordNetLemmatizer()

# --- NLTK-based Intent Classifier ---
class NLTKIntentClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.words = []
        self.classes = []
        self.intents = []

    def preprocess_text(self, text):
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return " ".join(words)

    def load_intents(self, filepath='intents.json'):
        with open(filepath, 'r') as file:
            self.intents = json.load(file)['intents']

        documents = []
        all_words = []
        self.classes = []

        for intent in self.intents:
            for pattern in intent['patterns']:
                processed_pattern = self.preprocess_text(pattern)
                documents.append((processed_pattern, intent['tag']))
                all_words.extend(processed_pattern.split())
            if intent['tag'] not in self.classes:
                self.classes.append(intent['tag'])

        self.words = sorted(list(set(all_words)))
        self.classes = sorted(list(set(self.classes)))

        return documents

    def train(self, documents):
        X_train = [doc[0] for doc in documents]
        y_train = [doc[1] for doc in documents]

        self.vectorizer = TfidfVectorizer()
        X_vectorized = self.vectorizer.fit_transform(X_train)

        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(X_vectorized, y_train)
        print("NLTK-based model trained successfully.")

    def predict_intent(self, sentence):
        processed_sentence = self.preprocess_text(sentence)
        vectorized_sentence = self.vectorizer.transform([processed_sentence])

        probabilities = self.model.predict_proba(vectorized_sentence)[0]
        predicted_class_index = np.argmax(probabilities)
        predicted_class = self.model.classes_[predicted_class_index]
        confidence = probabilities[predicted_class_index]

        return predicted_class, confidence

    def save_model(self, filepath='nltk_model.pkl'):
        with open(filepath, 'wb') as file:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'model': self.model,
                'words': self.words,
                'classes': self.classes,
                'intents': self.intents
            }, file)
        print(f"NLTK model saved to {filepath}")

    def load_model(self, filepath='nltk_model.pkl'):
        try:
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
                self.vectorizer = data['vectorizer']
                self.model = data['model']
                self.words = data['words']
                self.classes = data['classes']
                self.intents = data['intents']
            print(f"NLTK model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"NLTK model not found at {filepath}. Please train it first.")
            return False

    def get_response(self, intent_tag):
        for intent in self.intents:
            if intent['tag'] == intent_tag:
                return random.choice(intent['responses'])
        return "I'm sorry, I don't understand."


# --- Hugging Face Transformers Intent Classifier ---
class TransformerIntentClassifier:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.model_name = model_name
        self.intents_data = []
        self.id2label = {}
        self.label2id = {}

    def load_intents(self, filepath='intents.json'):
        with open(filepath, 'r') as file:
            self.intents_data = json.load(file)['intents']

        labels = sorted(list(set([intent['tag'] for intent in self.intents_data])))
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.label2id = {label: i for i, label in enumerate(labels)}
        print(f"Loaded {len(labels)} unique intent labels.")

        return self.intents_data

    def prepare_dataset(self, intents_data):
        texts, labels = [], []
        for intent in intents_data:
            for pattern in intent['patterns']:
                texts.append(pattern)
                labels.append(self.label2id[intent['tag']])

        features = Features({'text': Value('string'), 'label': Value('int64')})
        dataset = Dataset.from_dict({'text': texts, 'label': labels}, features=features)

        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length")

        return dataset.map(tokenize_function, batched=True)

    def train(self, train_dataset, model_output_dir="./transformer_model"):
        if os.path.exists(model_output_dir) and os.listdir(model_output_dir):
            print(f"Model already exists in {model_output_dir}. Loading instead of retraining...")
            self.load_model(model_output_dir)
            return

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.label2id), id2label=self.id2label, label2id=self.label2id
        )

        training_args = TrainingArguments(
            output_dir=model_output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        self.model.save_pretrained(model_output_dir)
        self.tokenizer.save_pretrained(model_output_dir)
        print(f"Transformer model saved to {model_output_dir}")

    def load_model(self, filepath="./transformer_model"):
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(filepath)
            self.tokenizer = AutoTokenizer.from_pretrained(filepath)

            if hasattr(self.model.config, 'id2label'):
                self.id2label = self.model.config.id2label
                self.label2id = {v: k for k, v in self.id2label.items()}
            else:
                self.load_intents()

            print(f"Transformer model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading Transformer model from {filepath}: {e}")
            return False

    def predict_intent(self, sentence):
        if not self.model or not self.tokenizer:
            print("Model not loaded. Call load_model() first.")
            return None, 0.0

        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

        predicted_class_id = np.argmax(probabilities)
        predicted_tag = self.id2label[predicted_class_id]
        confidence = probabilities[predicted_class_id]

        return predicted_tag, confidence

    def get_response(self, intent_tag):
        for intent in self.intents_data:
            if intent['tag'] == intent_tag:
                return random.choice(intent['responses'])
        return "I'm sorry, I don't have a response for that."
