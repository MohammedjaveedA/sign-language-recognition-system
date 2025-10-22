import cv2
import numpy as np
import pickle
import mediapipe as mp
from collections import deque
import time
import pyttsx3
import threading
from queue import Queue
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import base64
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import eventlet

eventlet.monkey_patch()

# CNN model definition
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageCNN, self).__init__()
        self.fc1 = nn.Linear(63, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Wrapper class to make CNN behave like RandomForest
class CNNWrapper:
    def __init__(self, model, label_encoder, device):
        self.model = model
        self.label_encoder = label_encoder
        self.device = device
        self.classes_ = label_encoder.classes_
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_labels = [self.label_encoder.classes_[idx] for idx in predicted.cpu().numpy()]
        return predicted_labels
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()

class RealTimeSignRecognizer:
    def __init__(self, model_path="sign_language_model.pkl"):
        print("🚀 Initializing RealTimeSignRecognizer...")
        
        # Load the trained model
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✅ Model loaded successfully!")
            print(f"✅ Model classes: {self.model.classes_}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            return
        
        # Initialize MediaPipe with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Only 1 hand for better performance
            min_detection_confidence=0.6,  # Lower for faster detection
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Recognition settings
        self.prediction_buffer = deque(maxlen=5)  # Smaller buffer for faster response
        self.last_prediction = ""
        self.prediction_confidence = 0.0
        self.last_detection_time = time.time()
        
        # Camera and processing
        self.cap = None
        self.is_processing = False
        self.processing_thread = None
        self.frame_queue = Queue(maxsize=2)  # Small queue to prevent lag
        
        # Performance optimization
        self.frame_skip = 2  # Process every 2nd frame (15 FPS processing)
        self.frame_counter = 0
        self.last_frame_time = 0
        
        # Text-to-speech
        self.tts_engine = None
        self.init_tts()
        
        print("✅ RealTimeSignRecognizer initialized successfully!")
    
    def init_tts(self):
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
        except:
            self.tts_engine = None
    
    def speak_text(self, text):
        if self.tts_engine and text:
            def speak():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except:
                    pass
            threading.Thread(target=speak, daemon=True).start()
    
    def extract_hand_landmarks(self, frame):
        """Optimized landmark extraction"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            landmarks = []
            hands_detected = False
            
            if results.multi_hand_landmarks:
                hands_detected = True
                # Use only the first hand for performance
                hand_landmarks = results.multi_hand_landmarks[0]
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Pad with zeros if no hands detected
            while len(landmarks) < 63:
                landmarks.extend([0.0, 0.0, 0.0])
            
            return landmarks[:63], hands_detected, results
            
        except Exception as e:
            print(f"Landmark extraction error: {e}")
            return [0.0] * 63, False, None
    
    def get_smoothed_prediction(self, prediction, confidence):
        """Fast prediction smoothing"""
        self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) >= 3:  # Reduced from 5 to 3
            recent_predictions = [pred for pred, conf in list(self.prediction_buffer)[-3:]]
            
            # Simple majority voting
            pred_counts = {}
            for pred in recent_predictions:
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
            
            most_common_pred = max(pred_counts, key=pred_counts.get)
            count = pred_counts[most_common_pred]
            
            if count >= 2 and confidence > 0.4:  # Lower confidence threshold
                if most_common_pred != self.last_prediction:
                    self.speak_text(most_common_pred)
                return most_common_pred, confidence
        
        return self.last_prediction, self.prediction_confidence
    
    def camera_capture_loop(self):
        """Fast camera capture in separate thread"""
        print("📷 Starting camera capture...")
        start_time = time.time()
        
        # Try different camera backends for faster startup
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]  # Windows, Windows Media, Any
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(0, backend)
                if self.cap.isOpened():
                    print(f"✅ Camera opened with backend: {backend}")
                    break
            except:
                continue
        
        if not self.cap or not self.cap.isOpened():
            print("❌ Failed to open camera with any backend")
            return False
        
        # Set camera properties for fast startup
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Single buffer to reduce latency
        
        print(f"⏱️ Camera started in {time.time() - start_time:.2f} seconds")
        
        while self.is_processing:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Limit queue size to prevent lag
            if self.frame_queue.qsize() < 2:
                self.frame_queue.put(frame)
            else:
                # Drop frame if queue is full (prevent memory buildup)
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except:
                    pass
            
            time.sleep(0.001)  # Small sleep to prevent CPU overload
    
    def processing_loop(self, socketio, sid):
        """Fast processing loop"""
        print("⚡ Starting processing loop...")
        
        last_sent_time = 0
        min_frame_interval = 1.0 / 15  # 15 FPS max to reduce lag
        
        while self.is_processing:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(timeout=1.0)
                
                current_time = time.time()
                if current_time - last_sent_time < min_frame_interval:
                    # Skip frame to maintain target FPS
                    continue
                
                # Process frame
                landmarks, hands_detected, results = self.extract_hand_landmarks(frame)
                
                # Draw landmarks if detected
                if results and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )
                
                # Make prediction
                current_prediction = ""
                current_confidence = 0.0
                
                if hands_detected and any(landmarks) and any(np.array(landmarks) != 0):
                    try:
                        prediction_probs = self.model.predict_proba([landmarks])[0]
                        predicted_class_idx = np.argmax(prediction_probs)
                        predicted_class = self.model.classes_[predicted_class_idx]
                        confidence = prediction_probs[predicted_class_idx]
                        
                        current_prediction, current_confidence = self.get_smoothed_prediction(
                            predicted_class, confidence
                        )
                        
                        if confidence > 0.4:  # Lower threshold for faster response
                            self.last_prediction = current_prediction
                            self.prediction_confidence = current_confidence
                            self.last_detection_time = time.time()
                            
                    except Exception as e:
                        print(f"Prediction error: {e}")
                
                # Use recent prediction
                if time.time() - self.last_detection_time < 1.5:  # Shorter memory
                    current_prediction = self.last_prediction
                    current_confidence = self.prediction_confidence
                
                # Add prediction overlay to frame
                if current_prediction:
                    cv2.putText(frame, f"Sign: {current_prediction}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {current_confidence:.2f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Show hand to camera", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Compress frame for faster transmission
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # Lower quality for speed
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Send results
                result_data = {
                    'prediction': current_prediction,
                    'confidence': float(current_confidence),
                    'hands_detected': hands_detected,
                    'frame': f"data:image/jpeg;base64,{frame_data}",
                    'timestamp': time.time()
                }
                
                socketio.emit('frame_processed', result_data, room=sid)
                last_sent_time = current_time
                
            except Exception as e:
                if self.is_processing:  # Only log if we're supposed to be running
                    print(f"Processing error: {e}")
                continue
        
        print("⏹️ Processing loop stopped")
    
    def start_processing(self, socketio, sid):
        """Start camera and processing with fast initialization"""
        if self.is_processing:
            return False
        
        print("🎬 Starting real-time processing...")
        self.is_processing = True
        
        # Clear previous data
        self.prediction_buffer.clear()
        self.last_prediction = ""
        self.prediction_confidence = 0.0
        
        # Start camera capture in separate thread
        camera_thread = threading.Thread(target=self.camera_capture_loop, daemon=True)
        camera_thread.start()
        
        # Wait a moment for camera to initialize
        time.sleep(0.5)
        
        # Start processing loop
        self.processing_thread = threading.Thread(
            target=self.processing_loop, 
            args=(socketio, sid),
            daemon=True
        )
        self.processing_thread.start()
        
        return True
    
    def stop_processing(self):
        """Fast cleanup"""
        print("⏹️ Stopping processing...")
        self.is_processing = False
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print("✅ Processing stopped")

# Flask App
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   async_mode='eventlet',
                   max_http_buffer_size=10 * 1024 * 1024,  # 10MB buffer
                   ping_timeout=60,
                   ping_interval=25)

# Import translation service
from translation_service import translation_service

# Global recognizer instance
recognizer = RealTimeSignRecognizer()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if recognizer and recognizer.model else 'unhealthy',
        'model_loaded': recognizer is not None and recognizer.model is not None,
        'classes': recognizer.model.classes_.tolist() if recognizer and recognizer.model else []
    })

# Translation API endpoints
@app.route('/api/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages for translation"""
    return jsonify({
        'languages': translation_service.get_supported_languages(),
        'current_language': translation_service.target_language
    })

@app.route('/api/set-language', methods=['POST'])
def set_language():
    """Set the target language for translation"""
    data = request.get_json()
    language_code = data.get('language', 'en')
    
    success = translation_service.set_target_language(language_code)
    
    return jsonify({
        'success': success,
        'language': language_code if success else translation_service.target_language,
        'language_name': translation_service.supported_languages.get(
            language_code if success else translation_service.target_language, 
            'English'
        )
    })

@app.route('/api/translate', methods=['POST'])
def translate_text():
    """Translate text to target language"""
    data = request.get_json()
    text = data.get('text', '')
    
    translated_text, source_lang = translation_service.translate_text(text)
    
    return jsonify({
        'original_text': text,
        'translated_text': translated_text,
        'source_language': source_lang,
        'target_language': translation_service.target_language
    })

@socketio.on('connect')
def handle_connect():
    print('✅ Client connected:', request.sid)
    emit('connected', {'status': 'connected', 'message': 'Ready for sign recognition'})

@socketio.on('disconnect')
def handle_disconnect():
    print('❌ Client disconnected:', request.sid)
    if recognizer:
        recognizer.stop_processing()

@socketio.on('start_recognition')
def handle_start_recognition():
    print('🎬 Client requested recognition start')
    
    if not recognizer or not recognizer.model:
        emit('error', {'message': 'Recognizer not initialized'})
        return
    
    if recognizer.is_processing:
        emit('error', {'message': 'Recognition already running'})
        return
    
    # Start processing
    success = recognizer.start_processing(socketio, request.sid)
    
    if success:
        emit('recognition_started', {'status': 'started'})
        print('✅ Recognition started successfully')
    else:
        emit('error', {'message': 'Failed to start recognition'})

@socketio.on('stop_recognition')
def handle_stop_recognition():
    print('⏹️ Client requested recognition stop')
    if recognizer:
        recognizer.stop_processing()
    emit('recognition_stopped', {'status': 'stopped'})

@socketio.on('speak_text')
def handle_speak_text(data):
    text = data.get('text', '')
    if recognizer and text:
        recognizer.speak_text(text)
        print(f"🔊 Speaking: {text}")

@socketio.on('set_language')
def handle_set_language(data):
    """Handle language change via WebSocket"""
    language_code = data.get('language', 'en')
    success = translation_service.set_target_language(language_code)
    
    emit('language_changed', {
        'success': success,
        'language': language_code if success else translation_service.target_language,
        'language_name': translation_service.supported_languages.get(
            language_code if success else translation_service.target_language, 
            'English'
        )
    })

@socketio.on('translate_text')
def handle_translate_text(data):
    """Handle text translation via WebSocket"""
    text = data.get('text', '')
    
    translated_text, source_lang = translation_service.translate_text(text)
    
    emit('translation_result', {
        'original_text': text,
        'translated_text': translated_text,
        'source_language': source_lang,
        'target_language': translation_service.target_language
    })

if __name__ == '__main__':
    print("🚀 Starting Optimized Sign Language Recognition Server...")
    DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'en')
    print(f"🌐 Default language: {DEFAULT_LANGUAGE}")
    
    if recognizer and recognizer.model:
        print("✅ Server is ready!")
        print("✅ Model classes:", recognizer.model.classes_)
        print("🌐 WebSocket server running on: http://localhost:5000")
        print("⚡ Optimized for fast startup and smooth performance")
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000, 
                    debug=False, 
                    use_reloader=False,
                    log_output=False)
    else:
        print("❌ Server failed to initialize!")