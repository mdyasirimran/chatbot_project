import os
import json
from model import NLTKIntentClassifier, TransformerIntentClassifier

# Choose which model to train: 'nltk' or 'transformer'
MODEL_TYPE = 'transformer' # Change to 'nltk' to use the NLTK-based model

if __name__ == "__main__":
    intents_file = 'intents.json'
    
    if MODEL_TYPE == 'nltk':
        classifier = NLTKIntentClassifier()
        documents = classifier.load_intents(filepath=intents_file)
        classifier.train(documents)
        classifier.save_model()
    elif MODEL_TYPE == 'transformer':
        # Ensure the model directory exists
        model_output_dir = "./transformer_model"
        os.makedirs(model_output_dir, exist_ok=True)

        classifier = TransformerIntentClassifier()
        intents_data = classifier.load_intents(filepath=intents_file)
        
        # Prepare the dataset for training
        train_dataset = classifier.prepare_dataset(intents_data)
        
        # Train the model
        classifier.train(train_dataset, model_output_dir=model_output_dir)
    else:
        print("Invalid MODEL_TYPE specified. Choose 'nltk' or 'transformer'.")

    print(f"Model training complete for {MODEL_TYPE} model.")