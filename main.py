#main.py 

import os
import openai
from flask import Flask, render_template, request, jsonify
import bleach
import json
import logging
from datetime import datetime

app = Flask(__name__)
openai.api_key = os.environ['OPENAI_API_KEY']

# Configure logging
logging.basicConfig(filename='logs/app.log', level=logging.INFO)

# Ensure the conversations directory exists
conversations_dir = os.path.join('static', 'conversations')
if not os.path.exists(conversations_dir):
    os.makedirs(conversations_dir)


@app.route('/')
def index():
    logging.info('Rendering index page')
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    user_message_cleaned = bleach.clean(user_message, strip=True)

    if len(user_message_cleaned) > 3000:
        logging.warning('User message exceeds 3000 characters')
        return jsonify({'error': 'Message too long. Max 3000 characters.'})
    with open('static/instructions.md', 'r') as f:
        instructions = f.read()

    with open('static/navajo_dictionary.md', 'r') as f:
        additional_instructions = f.read()

    system_prompt = f"{instructions}\n\nAdditional Instructions:\n{additional_instructions}"

    messages = [        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message_cleaned}
    ]

    temperature = float(request.form.get('temperature', 1.0))
    top_p = float(request.form.get('top_p', 1.0))
    max_tokens = int(request.form.get('max_tokens', 150))

    logging.info('Sending request to OpenAI API')
    completion = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    assistant_message = completion.choices[0].message.content
    logging.info('Received response from OpenAI API')

    # Store the conversation with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    conversation_filename = os.path.join(conversations_dir, f'conversation_{timestamp}.json')
    with open(conversation_filename, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "user_input": user_message_cleaned,
            "assistant_reply": assistant_message
        }, f)

    return jsonify({'message': assistant_message})


@app.route('/make-public', methods=['POST'])
def make_public():
  conversation = request.get_json()

  filename = 'static/conversation.json'
  with open(filename, 'w') as f:
    json.dump(conversation, f)

  logging.info('Conversation made public')
  return jsonify({'url': '/conversation'})


@app.route('/conversation')
def conversation():
  with open('static/conversation.json', 'r') as f:
    conversation = json.load(f)

  logging.info('Rendering conversation page')
  return render_template('conversation.html', conversation=conversation)


@app.route('/file')
def file():
  with open('static/navajo_dictionary.md', 'r') as f:
    content = f.read()

  logging.info('Rendering file page')
  return render_template('file.html', content=content)


def test_index():
  with app.test_client() as client:
    response = client.get('/')
    assert response.status_code == 200
    assert b'Welcome to the chatbot!' in response.data


def test_chat():
  with app.test_client() as client:
    response = client.post('/chat', data={'message': 'Hello'})
    assert response.status_code == 200
    assert 'message' in response.json


def test_chat_long_message():
  with app.test_client() as client:
    long_message = 'A' * 3001
    response = client.post('/chat', data={'message': long_message})
    assert response.status_code == 200
    assert 'error' in response.json


def test_make_public():
  with app.test_client() as client:
    conversation = [{'role': 'user', 'content': 'Hello'}]
    response = client.post('/make-public', json=conversation)
    assert response.status_code == 200
    assert 'url' in response.json


def test_conversation():
  with app.test_client() as client:
    response = client.get('/conversation')
    assert response.status_code == 200
    assert b'Public Conversation' in response.data


def test_file():
  with app.test_client() as client:
    response = client.get('/file')
    assert response.status_code == 200
    assert b'System Prompt' in response.data


if __name__ == '__main__':
  logging.info('Running unit tests')
  test_index()
  test_chat()
  test_chat_long_message()
  test_make_public()
  test_conversation()
  test_file()
  logging.info('Unit tests completed')

  app.run(host='0.0.0.0', port=81)
