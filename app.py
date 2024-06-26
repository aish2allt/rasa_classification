import os
import uuid
import threading
import logging
import flask
import rasa
from flask import Flask, request, jsonify
from typing import Text, Dict, Any
import nest_asyncio
from rasa.core.agent import Agent
from rasa.shared.utils.io import json_to_string
from rasa.shared.nlu.training_data.loading import load_data
from rasa.nlu.test import run_evaluation
from rasa.model_training import train_nlu
from rasa.model import get_local_model
import time
import asyncio

nest_asyncio.apply()

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

models = {}

# Default paths
DEFAULT_CONFIG_PATH = "C:\\Users\\aishw\\Rasa\\.venv\\config.yml"
DEFAULT_OUTPUT_PATH = "C:\\Users\\aishw\\Rasa\\.venv\\output"

def train_nlu_model(config_path: Text, training_data_path: Text, output_path: Text) -> Text:
    try:
        model_directory = train_nlu(
            config=config_path,
            nlu_data=training_data_path,
            output=output_path
        )
        
        model_path = get_local_model(model_directory)
        return model_path
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

async def load_and_predict(model_path: Text, example: Text) -> Dict[Text, Any]:
    try:
        agent = Agent.load(model_path)
        result = await agent.parse_message(example)
        return result
    except Exception as e:
        logger.error(f"Error loading and predicting: {e}")
        raise

def run_prediction(model_path: Text, example: Text) -> Dict[Text, Any]:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(load_and_predict(model_path, example))
        return result
    except Exception as e:
        logger.error(f"Error running prediction: {e}")
        raise
    
def train_model(model_id, config_path, training_data_path, output_path, fine_tune, model_path=None):
    try:
        model_path = train_nlu_model(config_path, training_data_path, output_path)
        models[model_id]['model_path'] = model_path
        models[model_id]['status'] = 'TRAINED'
        logger.info(f"Model {model_id} trained successfully. Model path: {model_path}")
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
    config_path = data.get('config_path', DEFAULT_CONFIG_PATH)
    training_data_path = data['training_data_path']
    output_path = data.get('output_path', DEFAULT_OUTPUT_PATH)
    fine_tune = data.get('fine_tune', False)
    model_path = data.get('model_path', None)

    models[model_id] = {
        'model_name': model_name,
        'status': 'IN_TRAINING',
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    }

    threading.Thread(target=train_model, args=(model_id, config_path, training_data_path, output_path, fine_tune, model_path)).start()
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

    model = models.get(model_id)
    if not model:
        return jsonify({'error': 'Model not found'}), 404
    if model['status'] == 'IN_TRAINING':
        return jsonify({'error': 'Model is currently in training'}), 400

    model_path = model.get('model_path')
    if not model_path:
        return jsonify({'error': "'model_path' not set"}), 500

    try:
        result = run_prediction(model_path, message)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()