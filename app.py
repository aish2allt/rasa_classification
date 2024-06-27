import os
import uuid
import threading
import logging
import flask
import rasa
from flask import Flask, request, jsonify
from typing import Text, Dict, Any, List, Optional, Union
import nest_asyncio
from rasa.core.agent import Agent
from rasa.shared.utils.io import json_to_string
from rasa.shared.nlu.training_data.loading import load_data
from rasa.nlu.test import run_evaluation
from rasa.model_training import train_nlu
from rasa.model import get_local_model
from rasa.shared.core.domain import Domain
import time
import asyncio
import yaml
import json
import tempfile
import random

nest_asyncio.apply()

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

models = {}

# Default paths
DEFAULT_CONFIG_PATH = r"C:\Users\aishw\Rasa\.venv\config.yml"
DEFAULT_OUTPUT_PATH = r"C:\Users\aishw\Rasa\.venv\output1"

def convert_json_to_yaml(json_data):
    yaml_data = {
        "version": "2.0",
        "nlu": []
    }
    for intent_class in json_data["intent_classes"]:
        intent_data = {
            "intent": intent_class["intent_name"],
            "examples": "\n".join([f"- {phrase}" for phrase in intent_class["training_phrases"]])
        }
        yaml_data["nlu"].append(intent_data)
    return yaml.dump(yaml_data, sort_keys=False)

def augment_training_data(training_phrases: List[str], num_augmented: int = 5) -> List[str]:
    augmented_phrases = []
    for phrase in training_phrases:
        words = phrase.split()
        for _ in range(num_augmented):
            new_phrase = ' '.join(random.sample(words, len(words)))
            augmented_phrases.append(new_phrase)
    return augmented_phrases

def train_nlu_model(config_path: Text, training_data: Dict, output_path: Text, fine_tune: bool = False, model_to_finetune: Text = None) -> Text:
    try:
        for intent_class in training_data["intent_classes"]:
            intent_class["training_phrases"].extend(
                augment_training_data(intent_class["training_phrases"])
            )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
            yaml_content = convert_json_to_yaml(training_data)
            temp_file.write(yaml_content)
            temp_file_path = temp_file.name

        model_directory = train_nlu(
            config=config_path,
            nlu_data=temp_file_path,
            output=output_path,
            fixed_model_name=None,
            persist_nlu_training_data=False,
            additional_arguments=None,
            domain=None,
            model_to_finetune=model_to_finetune if fine_tune else None,
            finetuning_epoch_fraction=1.0 if fine_tune else None
        )
        
        os.unlink(temp_file_path)  # Delete the temporary file
        
        return model_directory
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

async def load_and_predict(model_path: Text, example: Text, confidence_threshold: float = 0.7, fallback_intent: str = 'nlu_fallback') -> Dict[Text, Any]:
    try:
        agent = Agent.load(model_path)
        result = await agent.parse_message(example)
        
        if result['intent']['confidence'] < confidence_threshold:
            result['intent']['name'] = fallback_intent
            result['intent']['confidence'] = 1.0 - result['intent']['confidence']
        
        return result
    except Exception as e:
        logger.error(f"Error loading and predicting: {e}")
        raise

def run_prediction(model_path: Text, example: Text, confidence_threshold: float = 0.7, fallback_intent: str = 'nlu_fallback') -> Dict[Text, Any]:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(load_and_predict(model_path, example, confidence_threshold, fallback_intent))
        return result
    except Exception as e:
        logger.error(f"Error running prediction: {e}")
        raise
    
def train_model(model_id, config_path, training_data, output_path, fine_tune, model_to_finetune=None):
    try:
        model_directory = train_nlu_model(config_path, training_data, output_path, fine_tune, model_to_finetune)
        models[model_id]['model_path'] = model_directory
        models[model_id]['status'] = 'TRAINED'
        logger.info(f"Model {model_id} trained successfully. Model directory: {model_directory}")
    except Exception as e:
        models[model_id]['status'] = 'ERROR'
        models[model_id]['error'] = str(e)
        logger.error(f"Error training model {model_id}: {e}")

@app.route('/')
def index():
    return "Welcome to the Rasa NLU Model Training and Inference API!"

@app.route('/train', methods=['POST'])
def create_model():
    data = request.json
    model_id = str(uuid.uuid4())
    model_name = data.get('model_name', f"model_{model_id}")
    config_path = DEFAULT_CONFIG_PATH
    training_data = data
    output_path = DEFAULT_OUTPUT_PATH
    fine_tune = data.get('fine_tune', False)
    model_to_finetune = None

    if fine_tune:
        model_to_finetune_id = data.get('model_to_finetune_id')
        if model_to_finetune_id:
            model_to_finetune = models.get(model_to_finetune_id, {}).get('model_path')
            if not model_to_finetune:
                return jsonify({'error': 'Model to fine-tune not found or not trained'}), 400
        else:
            return jsonify({'error': 'model_to_finetune_id is required for fine-tuning'}), 400

    models[model_id] = {
        'model_name': model_name,
        'status': 'IN_TRAINING',
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    }

    threading.Thread(target=train_model, args=(model_id, config_path, training_data, output_path, fine_tune, model_to_finetune)).start()
    return jsonify({'model_id': model_id, 'start_time': models[model_id]['start_time']}), 201

@app.route('/status', methods=['GET'])
def status_check():
    model_id = request.args.get('model_id')
    model = models.get(model_id)
    if not model:
        return jsonify({'error': 'Model not found'}), 404
    
    return jsonify({'status': model['status']}), 200

@app.route('/infer', methods=['GET'])
def classify_utterance():
    model_id = request.args.get('model_id')
    message = request.args.get('message')
    confidence_threshold = float(request.args.get('confidence_threshold', 0.7))
    fallback_intent = request.args.get('fallback_intent', 'nlu_fallback')

    model = models.get(model_id)
    if not model:
        return jsonify({'error': 'Model not found'}), 404
    if model['status'] == 'IN_TRAINING':
        return jsonify({'error': 'Model is currently in training'}), 400

    model_path = model.get('model_path')
    if not model_path:
        return jsonify({'error': "'model_path' not set"}), 500

    try:
        result = run_prediction(model_path, message, confidence_threshold, fallback_intent)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()