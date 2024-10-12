import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import numpy as np
import pickle
import os

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model

# Tải mô hình trích xuất đặc trưng InceptionV3
@st.cache_resource
def load_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    model = Model(base_model.input, base_model.layers[-2].output)
    return model

def extract_features(image):
    feature_extractor = load_feature_extractor()
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = feature_extractor.predict(image, verbose=0)
    return feature

# Cập nhật hàm generate_caption để sử dụng trích xuất đặc trưng
def generate_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    image = extract_features(image)  # Trích xuất đặc trưng từ ảnh
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Đảm bảo các đầu vào có kiểu dữ liệu đúng
        sequence = sequence.astype('float32')
        
        # Sử dụng mô hình để dự đoán
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text



# Ứng dụng Streamlit
st.title("Image Caption Generator")

# Tải hình ảnh từ người dùng
uploaded_file = st.file_uploader("Chọn một hình ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị hình ảnh đã tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption='Hình ảnh đã tải lên.', use_column_width=True)
    
    # Tải mô hình và tokenizer
    model = load_inception_model()
    tokenizer = load_tokenizer()
    
    if model is not None and tokenizer is not None:
        # Tiền xử lý hình ảnh
        img = image.resize((299, 299))  # InceptionV3 cần ảnh kích thước 299x299
        img = np.array(img)
        caption = generate_caption(model, img, tokenizer, max_length=34)
        st.write("**Mô tả hình ảnh:**", caption)

        # Sinh mô tả
        caption = generate_caption(model, img, tokenizer, max_length=34)
        st.write("**Mô tả hình ảnh:**", caption)
