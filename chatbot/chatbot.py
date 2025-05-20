import random
import pickle
import json
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

# Initialize
lemmatizer = WordNetLemmatizer()

# Load resources
intents = json.load(open('D:\Soft Sparks\Chatbot Project\chatbot\Include\intents.json'))  # Use forward slashes
words = pickle.load(open('D:\Soft Sparks\Chatbot Project\models\words.pkl', 'rb'))
classes = pickle.load(open('D:\Soft Sparks\Chatbot Project\models\classes.pkl', 'rb'))
model = tf.keras.models.load_model("D:\Soft Sparks\Chatbot Project\models\chatbot_model.h5")

# Symptom keywords
symptoms = ['head', 'headache', 'fever', 'stomach', 'stomach pain', 'vomit', 'vomiting', 'diarrhea', 'eye', 'eye pain', 'pain killer']

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    query = bag_of_words(sentence)
    res = model.predict(np.array([query]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return [{'intent': 'fallback', 'probability': '1.0'}]

    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def has_multiple_symptoms(sentence):
    count = 0
    sentence_lower = sentence.lower()
    for symptom in symptoms:
        if symptom in sentence_lower:
            count += 1
    return count > 1

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tags'] == tag:
            return random.choice(i['response']), tag
    else:
        return "Sorry, I do not understand. Can you try again?", tag

def chatbot_response(message):
    if has_multiple_symptoms(message):
        intents_list = [{'intent': 'multiple_symptoms', 'probability': '1.0'}]
    else:
        intents_list = predict_class(message)
    response, tag = get_response(intents_list, intents)
    return response, tag


