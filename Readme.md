# 🌐 Sign Language Recognition System

A *real-time sign language recognition system* that uses *computer vision* and *machine learning* to detect hand signs and translate them into multiple languages.  

Built with *React (frontend), **Flask (backend), and **MediaPipe (hand tracking)*.

---

## 🚀 Features

- 🎥 *Real-time Sign Recognition* — Detects hand signs using a live webcam feed.  
- 🌍 *Multi-language Translation* — Translates recognized signs into multiple languages.  
- 🔊 *Text-to-Speech* — Speaks both the original and translated text aloud.  

---

## 🛠 Tech Stack

*Frontend*
- ⚛ React — Modern UI framework  
- 🔌 Socket.IO Client — Real-time communication  
- 🎨 CSS3 — Responsive styling & animations  

*Backend*
- 🐍 Flask — Python web framework  
- 🔄 Socket.IO — WebSocket communication  
- ✋ MediaPipe — Hand landmark detection  
- 📷 OpenCV — Computer vision processing  
- 🔥 PyTorch — Deep learning model  
- 🌐 Google Translate API — Translation service  

*Machine Learning*
- 🧠 CNN Model — Custom neural network for sign classification  
- ✋ MediaPipe Hands — Real-time hand tracking  
- 📊 Scikit-learn — Data preprocessing & evaluation  

---

## 📋 Prerequisites

Ensure you have the following installed:
- [Python 3.8+](https://www.python.org/downloads/)
- [Node.js 14+](https://nodejs.org/)
- [Git](https://git-scm.com/)
- A *webcam*

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/sign-language-recognition.git
cd sign-language-recognition

```

---


### 2️⃣ Backend Setup

Create Virtual Environment (Recommended)

Windows:
```bash

python -m venv sign_env
sign_env\Scripts\activate
```
macOS/Linux:
```bash
python3 -m venv sign_env
source sign_env/bin/activate
```
Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```
Environment Configuration

Create a .env file inside the backend/ directory:

# backend/.env
DEFAULT_LANGUAGE=en
FLASK_ENV=development
SERVER_HOST=0.0.0.0
SERVER_PORT=5000


---

### 3️⃣ Frontend Setup
```bash
cd frontend
npm install
```

---

### 4️⃣ Run the Application

Start Backend Server:
```bash
cd backend
python app.py
```
Server runs on → http://localhost:5000

Start Frontend Server:
```bash
cd frontend
npm run dev
```
Frontend runs on → http://localhost:3000


---
```bash
📁 Project Structure

sign-language-recognition/
├── backend/
│   ├── .env                     # Environment variables
│   ├── app.py                   # Main Flask application
│   ├── translation_service.py   # Multi-language translation logic
│   ├── train_model.py           # Model training script
│   ├── collect_data.py          # Data collection utility
│   ├── realtime_recognition.py  # Real-time recognition
│   ├── image_preprocessing.py   # Image enhancement
│   ├── requirements.txt         # Python dependencies
│   └── sign_language_model.pkl  # Trained CNN model
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main React component
│   │   └── App.css              # Styling
│   ├── package.json             # Node dependencies
│   └── public/
│
└── README.md
```

---

🎯 Usage Guide

▶ Starting Recognition

1. Click "Start Recognition" to begin the camera feed.


2. Ensure good lighting and clear hand visibility.


3. Keep your hand inside the camera frame.



🌍 Using Translation

1. Select target language from the dropdown.


2. Show a hand sign to the camera.


3. Click "Translate Current Sign" for instant translation.


4. Use "Speak" buttons for audio output.