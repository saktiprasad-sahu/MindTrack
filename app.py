import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from keras import backend as K

# -------------------- Clear TensorFlow session --------------------
def clear_session():
    K.clear_session()

# -------------------- Load Models and Tokenizer --------------------
if 'text_model' not in st.session_state:
    clear_session()  # Clear previous session to avoid issues
    st.session_state['text_model'] = load_model("text_emotion_model.h5")
    st.session_state['image_model'] = load_model("image_emotion_model.h5", compile=False)
    st.session_state['tokenizer'] = pickle.load(open("tokenizer.pkl", "rb"))
    st.session_state['label_encoder_text'] = pickle.load(open("label_encoder.pkl", "rb"))
    st.session_state['label_encoder_image'] = pickle.load(open("image_label_encoder.pkl", "rb"))

text_model = st.session_state['text_model']
image_model = st.session_state['image_model']
tokenizer = st.session_state['tokenizer']
label_encoder_text = st.session_state['label_encoder_text']
label_encoder_image = st.session_state['label_encoder_image']

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="MindTrack", layout="wide")
st.title("🧠 MindTrack: Multimodal Emotion Detection App")
st.markdown("Detect emotions from **Text**, **Images**, and **Emojis** using AI.")

# -------------------- Session State for Emotion History --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- Sidebar Navigation --------------------
option = st.sidebar.radio("Choose Detection Type", ["Text", "Image", "Emoji"])

# -------------------- Text Emotion Detection --------------------
if option == "Text":
    st.header("💬 Text-Based Emotion Detection")
    text_input = st.text_area("Enter your thoughts:")

    if st.button("Analyze Text"):
        if text_input.strip() != "":
            sequence = tokenizer.texts_to_sequences([text_input])
            padded = pad_sequences(sequence, maxlen=100)
            prediction = text_model.predict(padded).squeeze()
            predicted_index = np.argmax(prediction)
            emotion = label_encoder_text.classes_[predicted_index]

            st.success(f"**Predicted Emotion:** {emotion}")

            # Save to history
            st.session_state.history.append({
                "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Input Type": "Text",
                "Input": text_input,
                "Predicted Emotion": emotion
            })

            # Plotting
            fig, ax = plt.subplots()
            sns.barplot(x=prediction, y=label_encoder_text.classes_, ax=ax)
            ax.set_title("Text Emotion Probabilities")
            st.pyplot(fig)

# -------------------- Image Emotion Detection --------------------
elif option == "Image":
    st.header("🖼️ Image-Based Emotion Detection")
    uploaded_image = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("L")
        image = image.resize((48, 48))
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 48, 48, 1)

        prediction = image_model.predict(img_array).squeeze()
        predicted_index = np.argmax(prediction)
        emotion = label_encoder_image.classes_[predicted_index]

        st.image(uploaded_image, caption="Uploaded Face Image", width=150)
        st.success(f"**Predicted Emotion from Image:** {emotion}")

        # Save to history
        st.session_state.history.append({
            "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Input Type": "Image",
            "Input": "Face Image",
            "Predicted Emotion": emotion
        })

        fig, ax = plt.subplots()
        sns.barplot(x=prediction, y=label_encoder_image.classes_, ax=ax)
        ax.set_title("Image Emotion Probabilities")
        st.pyplot(fig)

# -------------------- Emoji Emotion Detection --------------------
elif option == "Emoji":
    st.header("😀 Emoji-Based Emotion Detection")

    emoji_emotion_map = {
        "😊": "Happy",
        "😂": "Joy",
        "😢": "Sad",
        "😠": "Angry",
        "😨": "Fear",
        "😮": "Surprise",
        "😴": "Boredom",
        "😎": "Confidence",
        "❤️": "Love",
        "😔": "Depressed"
    }

    emoji_choice = st.selectbox("Pick an emoji to analyze:", list(emoji_emotion_map.keys()))

    if emoji_choice:
        emotion = emoji_emotion_map.get(emoji_choice)
        st.success(f"**Detected Emotion:** {emotion}")

        # Save to history
        st.session_state.history.append({
            "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Input Type": "Emoji",
            "Input": emoji_choice,
            "Predicted Emotion": emotion
        })

# -------------------- Download History --------------------
if st.session_state.history:
    st.subheader("📥 Download Emotion History")
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history)

    csv = df_history.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv, file_name="emotion_history.csv", mime="text/csv")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("Made with ❤️ by Sakti Prasad Sahu | Deep Learning | NLP | Streamlit")
