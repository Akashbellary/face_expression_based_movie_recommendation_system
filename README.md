## 🚀 Demo

> Deployed on Streamlit Cloud  
> 🔗 Click this link, to try the deployed site : https://face-expression-based-movie-recommendation-system.streamlit.app/

---

# 🎭 Face Expression-Based Movie Recommendation System

A fun and smart web app that recommends movies based on your facial expressions using deep learning and computer vision!

---

## 🔍 About the Project

This project detects your current **facial emotion** using your webcam and recommends a list of movies that match your mood. It combines:

- 🎥 **Facial Emotion Recognition** using a trained CNN model
- 🎬 **Movie genre mapping** based on emotion
- 🌐 **Streamlit** for the frontend experience

---

## 🧠 How It Works

1. **Take a selfie** using the built-in camera tool
2. The app detects your **facial emotion** (like Happy, Sad, Angry, etc.)
3. It then recommends movies matching your emotion from a curated dataset

---

## 😄 Emotions & Mapped Genres

| Emotion   | Recommended Genres                         |
|-----------|---------------------------------------------|
| Angry     | Action, Thriller, Crime, War               |
| Disgust   | Mystery, Horror, Foreign                   |
| Fear      | Horror, Thriller, Mystery                  |
| Happy     | Comedy, Adventure, Animation, Music, Family|
| Sad       | Drama, Romance, History                    |
| Surprise  | Fantasy, Science Fiction, Animation        |
| Neutral   | Documentary, TV, Movie, Western            |

---

## 🛠 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io)
- **Model**: Pretrained CNN (Keras & TensorFlow)
- **Face Detection**: OpenCV Haar Cascades
- **Data**: Custom movie dataset with genres
- **Languages**: Python

---

## 📂 Folder Structure

