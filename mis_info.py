import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Constants
VOCAB_SIZE = 10000
MAX_LENGTH = 100

# Load model
model = load_model('model_nilesh.h5', compile=False)

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
