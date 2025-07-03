import streamlit as st
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

# 🎬 Streamlit Page Settings
st.set_page_config(page_title="🎭 Emotion Movie Recommender", layout="centered")
st.title("🎭 Facial Emotion-Based Movie Recommender")
st.markdown("📸 Upload or capture your face to get movie recommendations based on your expression!")

# 📦 Load FER Model
@st.cache_resource
def load_fer_model():
    return load_model("model/FER_model.h5")
model = load_fer_model()

# 📚 Load Movie Dataset
@st.cache_data
def load_movies():
    return pd.read_csv("movies.csv")
df = load_movies()

# 🎭 Emotion to Genre Mapping
emotion_genre_map = {
    'Angry': ['Action', 'Thriller', 'Crime', 'War'],
    'Disgust': ['Mystery', 'Horror', 'Foreign'],
    'Fear': ['Horror', 'Thriller', 'Mystery'],
    'Happy': ['Comedy', 'Adventure', 'Animation', 'Family', 'Music'],
    'Sad': ['Drama', 'Romance', 'History'],
    'Surprise': ['Fantasy', 'Science', 'Fiction', 'Animation'],
    'Neutral': ['Documentary', 'TV', 'Movie', 'Western']
}


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 📷 Upload or capture image
img_file = st.file_uploader("📤 Upload your image", type=["jpg", "jpeg", "png"])
st.markdown("or")
cam_img = st.camera_input("📸 Take a photo")

image = img_file if img_file else cam_img

if image:
    st.image(image, caption="Input Image", width=300)
    st.info("⏳ Analyzing your emotion...")

    # 🧠 Process image
    img = Image.open(image).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        st.warning("😕 No face detected. Try again with better lighting or a clear front-facing image.")
    else:
        # ✅ Visualize detection
        for (x, y, w, h) in faces:
            cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        st.image(img_np, caption="✅ Face Detected", channels="RGB")

        # 📊 Predict emotion on first face
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        roi = face.astype("float32") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi, verbose=0)[0]
        emotion = emotion_labels[np.argmax(preds)]
        confidence = np.max(preds)

        st.success(f"🧠 Emotion Detected: **{emotion}** ({confidence * 100:.2f}%)")
        st.subheader("🎬 Recommended Movies:")

        # 🎥 Filter movies by genre
        genres = emotion_genre_map.get(emotion, [])
        filtered = df[df['Movie_Genre'].apply(lambda g: any(genre in g for genre in genres))]

        if not filtered.empty:
            for _, row in filtered.sample(min(5, len(filtered))).iterrows():
                st.markdown(f"**🎞️ {row['Movie_Title']}**  \n*Genres:* {row['Movie_Genre']}")
        else:
            st.info("No movies found for your emotion. Try another expression!")

st.markdown("---")
st.caption("🧠 Built by Akash — Powered by Streamlit, OpenCV & Keras")
