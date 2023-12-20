from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

model = tf.keras.models.load_model("D:\myproject\lstm_model.h5")
tokenizer = joblib.load("D:\myproject\model_tokenizer.joblib")

max_sequence_length = model.input_shape[1]

def generate_recipe(seed_text, next_words, model, tokenizer, max_sequence_length):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_length, padding='pre')
        predicted_probabilities = model.predict(token_list, verbose=0)

        predicted_index = np.argmax(predicted_probabilities)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word

    return seed_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_recipe', methods=['POST'])
def generate_recipe_endpoint():
    try:
        input_resep = request.form['input_resep']
        generated_recipe = generate_recipe(input_resep, 20, model, tokenizer, max_sequence_length)

        response_json = {'generated_recipe': generated_recipe}
        return jsonify(response_json)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
