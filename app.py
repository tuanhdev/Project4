import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import numpy as np
import pickle
from googletrans import Translator

# Load the InceptionV3 model for feature extraction
@st.cache_resource
def load_inception_model():
    base_model = InceptionV3(weights='imagenet')
    return tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

# Load the captioning model
@st.cache_resource
def load_captioning_model():
    return load_model('inception_caption_model.h5')

# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

# Function to preprocess image and extract features
def extract_features(image, model):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((299, 299))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return model.predict(image)

# Function to sample next word with temperature
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Function to generate captions
def generate_caption(model, features, tokenizer, max_length=35, temperature=0.8):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        
        preds = model.predict([features, sequence], verbose=0)
        preds = preds.flatten()
        
        next_word_index = sample(preds, temperature)
        word = tokenizer.index_word.get(next_word_index)
        
        if word is None or word == 'endseq':
            break

        in_text += ' ' + word

        last_words = in_text.split()[-3:]
        if len(set(last_words)) == 1:
            break
            
    return in_text.replace('startseq', '').strip()

# Function to translate English caption to Vietnamese
def translate_caption(caption, src_lang='en', dest_lang='vi'):
    translator = Translator()
    translated = translator.translate(caption, src=src_lang, dest=dest_lang)
    return translated.text

# Streamlit App
st.title("Image Caption Generator")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Load models and tokenizer
        inception_model = load_inception_model()
        caption_model = load_captioning_model()
        tokenizer = load_tokenizer()

        # Extract features from the image
        features = extract_features(image, inception_model)

        # Generate caption
        caption = generate_caption(caption_model, features, tokenizer)
        st.write("**Image Caption (EN):**", caption)

        # Translate caption to Vietnamese
        translated_caption = translate_caption(caption)
        st.write("**Chú thích hình ảnh (VI):**", translated_caption)
    
    except Exception as e:
        st.error(f"Error processing the image: {e}")
