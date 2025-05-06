import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from keras.layers import Layer
import tensorflow as tf

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                 initializer='zeros')        
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# Constants
VOCAB_SIZE = 10000
MAX_LENGTH = 100

# Load model
model = load_model('model_nilesh.h5', custom_objects={'Attention': Attention})

# Load tokenizer
with open('tokenizer.json') as f:
    tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)

# Prediction function
def predict_misinformation(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    label = "Misinformation" if prediction >= 0.5 else "Genuine"
    return prediction, label

# Streamlit UI
st.title("🧠 Misinformation Detection using LSTM + Attention")
st.write("Enter a tweet to check if it's **Misinformation** or **Genuine**.")

user_input = st.text_area("Tweet Text", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prob, label = predict_misinformation(user_input)
        st.markdown(f"### 🏷️ Prediction: `{label}`")
        st.markdown(f"**Confidence Score:** `{prob:.4f}`")
