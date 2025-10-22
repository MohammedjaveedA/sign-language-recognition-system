🌐 Sign Language Recognition & Translation System
A real-time sign language recognition system that uses computer vision and machine learning to detect hand signs and translate them into multiple languages. Built with React frontend, Flask backend, and MediaPipe for hand tracking.

🚀 Features
Real-time Sign Recognition: Detect hand signs using webcam with live video feed

Multi-language Translation: Translate recognized signs into 15+ languages

Text-to-Speech: Speak both original and translated text


🛠 Tech Stack
Frontend
React - Modern UI framework

Socket.IO Client - Real-time communication

CSS3 - Responsive styling with animations

Backend
Flask - Python web framework

Socket.IO - WebSocket communication

MediaPipe - Hand landmark detection

OpenCV - Computer vision processing

PyTorch - Deep learning model

Google Translate API - Free translation service

Machine Learning
CNN Model - Custom neural network for sign classification

MediaPipe Hands - Real-time hand tracking

Scikit-learn - Data preprocessing and evaluation

📋 Prerequisites
Before you begin, ensure you have the following installed:

Python 3.8+ Download here

Node.js 14+ Download here

Webcam - For real-time sign detection

Git - For version control

🚀 Quick Start
1. Clone the Repository
bash
git clone https://github.com/your-username/sign-language-recognition.git
cd sign-language-recognition
2. Backend Setup
Create Virtual Environment (Recommended)
bash
# Windows
python -m venv sign_env
sign_env\Scripts\activate

# macOS/Linux
python3 -m venv sign_env
source sign_env/bin/activate
Install Python Dependencies
bash
cd backend
pip install -r requirements.txt
Environment Configuration
Create a .env file in the backend directory:

bash
# Backend/.env
DEFAULT_LANGUAGE=en
FLASK_ENV=development
SERVER_HOST=0.0.0.0
SERVER_PORT=5000
3. Frontend Setup
bash
cd frontend
npm install
4. Run the Application
Start Backend Server
bash
cd backend
python app.py
The backend will start on http://localhost:5000

Start Frontend Development Server
bash
cd frontend
npm run dev
The frontend will start on http://localhost:3000

📁 Project Structure
text
sign-language-recognition/
├── backend/
│   ├── .env                          # Environment variables
│   ├── app.py                        # Main Flask application
│   ├── translation_service.py        # Multi-language translation
│   ├── train_model.py               # Model training script
│   ├── collect_data.py              # Data collection utility
│   ├── realtime_recognition.py      # Real-time recognition
│   ├── image_preprocessing.py       # Image enhancement
│   ├── requirements.txt             # Python dependencies
│   └── sign_language_model.pkl      # Trained model
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Main React component
│   │   └── App.css                  # Styling
│   ├── package.json                 # Node dependencies
│   └── public/
└── README.md
🎯 Usage Guide
1. Starting Recognition
Click "Start Recognition" to begin camera feed

Ensure proper lighting and clear hand visibility

Position hand within camera frame

2. Using Translation
Select target language from dropdown

Show hand sign to camera

Click "Translate Current Sign" for instant translation

Use "Speak" buttons for audio output

3. Adding New Signs
Method 1: Using Data Collection Tool
bash
cd backend
python collect_data.py
Follow on-screen instructions to capture new sign images.

Method 2: Manual Dataset Creation
Create folder in sign_data/your_sign_name/

Add training images (JPG/PNG)

Retrain model: python train_model.py

4. Supported Languages
🇺🇸 English | 🇪🇸 Spanish | 🇫🇷 French | 🇩🇪 German

🇮🇹 Italian | 🇵🇹 Portuguese | 🇷🇺 Russian | 🇨🇳 Chinese

🇯🇵 Japanese | 🇰🇷 Korean | 🇦🇪 Arabic | 🇮🇳 Hindi

🇧🇩 Bengali | 🇮🇳 Tamil | 🇮🇳 Telugu | 🇮🇳 Malayalam

🔧 Advanced Configuration
Model Training
To retrain the model with new data:

bash
cd backend
python train_model.py
The system will:

Scan sign_data/ directory for training data

Extract hand landmarks using MediaPipe

Train CNN model with new classes

Save updated model automatically