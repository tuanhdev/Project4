import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import numpy as np
import pickle
import os

# Tải mô hình và tokenizer
@st.cache_resource
def load_inception_model():
    if os.path.exists('inception_caption_model.h5'):
        return load_model('inception_caption_model.h5')
    else:
        st.error("Model file not found. Please upload 'inception_caption_model.h5'.")
        return None

@st.cache_resource
def load_tokenizer():
    if os.path.exists('tokenizer.pkl'):
        with open('tokenizer.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        st.error("Tokenizer file not found. Please upload 'tokenizer.pkl'.")
        return None

# Hàm dự đoán mô tả hình ảnh
def generate_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Kiểm tra đầu vào của mô hình để phát hiện lỗi sớm
        st.write("Image input shape:", image.shape)
        st.write("Sequence input shape:", sequence.shape)

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
        img = image.resize((299, 299))  # InceptionV3 cần hình ảnh kích thước 299x299
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 127.5 - 1.0  # Chuẩn hóa cho InceptionV3

        # Sinh mô tả
        caption = generate_caption(model, img, tokenizer, max_length=34)
        st.write("**Mô tả hình ảnh:**", caption)
