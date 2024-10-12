import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import numpy as np
import pickle

# Load model and tokenizer
@st.cache_resource
def load_inception_model():
    return load_model('inception_caption_model.h5')

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

# Generate image caption
def generate_caption(model, image, tokenizer, max_length=34):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Streamlit App
st.title("Image Caption Generator")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess image
    model = load_inception_model()
    tokenizer = load_tokenizer()
    
    img = image.resize((299, 299))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 127.5 - 1.0  # Normalize for InceptionV3

    # Generate caption
    caption = generate_caption(model, img, tokenizer)
    st.write("**Image Caption:**", caption)
