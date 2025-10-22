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
import json
import os

# CNN model definition - SAME AS IN TRAINING
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

# CNN Wrapper - SAME AS IN TRAINING
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

class SignDictionary:
    """Sign dictionary with detailed instructions and animations"""
    
    def __init__(self):
        self.sign_database = {
            'hello': {
                'name': 'Hello',
                'description': 'A friendly greeting gesture',
                'instructions': [
                    "Raise your dominant hand to head level",
                    "Keep fingers together and straight",
                    "Palm facing forward toward the person",
                    "Wave gently from side to side 2-3 times",
                    "Maintain eye contact and smile"
                ],
                'tips': "Keep the movement smooth and gentle, not too fast",
                'common_mistakes': "Don't make the waving motion too large or aggressive"
            },
            '3': {
                'name': 'Number Three',
                'description': 'Number three in sign language',
                'instructions': [
                    "Hold up your dominant hand",
                    "Extend thumb, index finger, and middle finger",
                    "Keep ring finger and pinky folded down",
                    "Palm facing forward",
                    "Hold steady for 2-3 seconds"
                ],
                'tips': "Make sure only three fingers are clearly extended",
                'common_mistakes': "Don't let other fingers partially extend"
            },
            '2': {
                'name': 'Number Two',
                'description': 'Number two in sign language',
                'instructions': [
                    "Hold up your dominant hand",
                    "Extend index and middle finger in V shape",
                    "Keep other fingers folded down",
                    "Palm facing forward",
                    "Keep fingers slightly separated"
                ],
                'tips': "Make a clear V shape with your two fingers",
                'common_mistakes': "Don't make a peace sign - keep palm forward"
            },
            '1': {
                'name': 'Number One',
                'description': 'Number one in sign language',
                'instructions': [
                    "Hold up your dominant hand",
                    "Extend only your index finger",
                    "Keep all other fingers folded down",
                    "Point finger upward",
                    "Palm can face forward or to the side"
                ],
                'tips': "Keep the index finger straight and steady",
                'common_mistakes': "Don't let other fingers partially extend"
            },
            'Thank You': {
                'name': 'Thank You',
                'description': 'Express gratitude and appreciation',
                'instructions': [
                    "Place fingertips of dominant hand on your chin",
                    "Start with fingers touching your lips/chin area",
                    "Move hand forward and slightly downward",
                    "End with palm facing up toward the person",
                    "Maintain gentle facial expression"
                ],
                'tips': "The movement should be smooth and graceful",
                'common_mistakes': "Don't start too low on the chin or move too quickly"
            },
            'Yes': {
                'name': 'Yes',
                'description': 'Affirmative response',
                'instructions': [
                    "Make a fist with your dominant hand",
                    "Extend thumb upward (thumbs up gesture)",
                    "Keep fist at chest level",
                    "Nod the fist up and down like a head nod",
                    "Repeat motion 2-3 times"
                ],
                'tips': "The nodding motion should be clear and deliberate",
                'common_mistakes': "Don't just hold a static thumbs up - add the nodding motion"
            },
            'No': {
                'name': 'No',
                'description': 'Negative response',
                'instructions': [
                    "Hold up your dominant hand",
                    "Extend index and middle finger",
                    "Bring thumb to touch the two extended fingers",
                    "Open and close the fingers against thumb",
                    "Like a mouth opening and closing"
                ],
                'tips': "Make the opening/closing motion clear and crisp",
                'common_mistakes': "Don't just hold static fingers - show the closing motion"
            }
        }
    
    def get_sign_info(self, sign_name):
        """Get detailed information about a sign"""
        return self.sign_database.get(sign_name.lower(), None)
    
    def get_all_signs(self):
        """Get list of all available signs"""
        return list(self.sign_database.keys())

class PracticeSession:
    """Practice mode for learning specific signs"""
    
    def __init__(self):
        self.reset_session()
        
    def reset_session(self):
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self.start_time = time.time()
        self.session_history = []
        
    def record_attempt(self, target_sign, predicted_sign, confidence, success):
        """Record a practice attempt"""
        self.attempts += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
            
        self.session_history.append({
            'target': target_sign,
            'predicted': predicted_sign,
            'confidence': confidence,
            'success': success,
            'timestamp': time.time()
        })
        
    def get_stats(self):
        """Get practice session statistics"""
        duration = time.time() - self.start_time
        accuracy = (self.successes / self.attempts * 100) if self.attempts > 0 else 0
        
        return {
            'attempts': self.attempts,
            'successes': self.successes,
            'failures': self.failures,
            'accuracy': accuracy,
            'duration': duration
        }
        
    def get_feedback(self):
        """Get performance feedback"""
        stats = self.get_stats()
        
        if stats['accuracy'] >= 80:
            return "Excellent! You're mastering this sign!"
        elif stats['accuracy'] >= 60:
            return "Good progress! Keep practicing for better consistency."
        elif stats['accuracy'] >= 40:
            return "Getting better! Focus on the sign instructions."
        else:
            return "Keep trying! Check the sign dictionary for proper technique."

class EnhancedSignRecognition:
    def __init__(self, model_path="sign_language_model.pkl"):
        # Load the trained model
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully!")
        except FileNotFoundError:
            print(f"Model file {model_path} not found. Please train the model first using train_model.py")
            self.model = None
            return
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize components
        self.sign_dictionary = SignDictionary()
        self.practice_session = PracticeSession()
        
        # App modes
        self.current_mode = 'recognition'  # 'recognition', 'practice', 'dictionary'
        self.practice_target_sign = None
        self.dictionary_current_sign = 0
        
        # Speed adjustment
        self.recognition_speed = 'normal'  # 'slow', 'normal', 'fast'
        self.speed_settings = {
            'slow': {'buffer_size': 20, 'confidence_threshold': 0.3, 'stability_frames': 8},
            'normal': {'buffer_size': 10, 'confidence_threshold': 0.5, 'stability_frames': 3},
            'fast': {'buffer_size': 5, 'confidence_threshold': 0.6, 'stability_frames': 2}
        }
        
        # For smoothing predictions
        self.prediction_buffer = deque(maxlen=self.speed_settings[self.recognition_speed]['buffer_size'])
        self.last_prediction = ""
        self.prediction_confidence = 0.0
        self.last_detection_time = time.time()
        
        # Initialize Text-to-Speech
        self.init_tts()
        
        # Audio control variables
        self.last_audio_time = 0
        self.audio_cooldown = 3.0
        self.audio_queue = Queue()
        self.audio_thread = None
        self.audio_enabled = True
        self.start_audio_thread()
        
        # UI state
        self.show_instructions = True
        
    def init_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
            print("Text-to-Speech initialized successfully!")
        except Exception as e:
            print(f"Error initializing TTS: {e}")
            self.tts_engine = None
    
    def start_audio_thread(self):
        """Start the audio processing thread"""
        self.audio_thread = threading.Thread(target=self.audio_worker, daemon=True)
        self.audio_thread.start()
    
    def audio_worker(self):
        """Worker thread for processing audio requests"""
        while True:
            try:
                message = self.audio_queue.get(timeout=1)
                if message == "STOP":
                    break
                if self.tts_engine and self.audio_enabled:
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait()
                self.audio_queue.task_done()
            except:
                continue
    
    def speak_text(self, text, force=False):
        """Add text to speech queue with cooldown"""
        current_time = time.time()
        if not force and (current_time - self.last_audio_time) < self.audio_cooldown:
            return
        try:
            self.audio_queue.put_nowait(text)
            self.last_audio_time = current_time
        except:
            pass
    
    def change_speed_setting(self, new_speed):
        """Change recognition speed setting"""
        if new_speed in self.speed_settings:
            self.recognition_speed = new_speed
            settings = self.speed_settings[new_speed]
            self.prediction_buffer = deque(maxlen=settings['buffer_size'])
            print(f"Speed changed to: {new_speed}")
    
    def extract_hand_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks = []
        hands_detected = False
        
        if results.multi_hand_landmarks:
            hands_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        while len(landmarks) < 63:
            landmarks.extend([0.0, 0.0, 0.0])
        
        return landmarks[:63], hands_detected, results
    
    def get_smoothed_prediction(self, prediction, confidence):
        """Smooth predictions using speed-adjusted buffer"""
        settings = self.speed_settings[self.recognition_speed]
        self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) >= settings['stability_frames']:
            recent_predictions = [pred for pred, conf in list(self.prediction_buffer)[-5:]]
            pred_counts = {}
            for pred in recent_predictions:
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
            
            most_common_pred = max(pred_counts, key=pred_counts.get)
            
            if (pred_counts[most_common_pred] >= settings['stability_frames'] and 
                confidence > settings['confidence_threshold']):
                
                if (most_common_pred != self.last_prediction and 
                    confidence > settings['confidence_threshold'] + 0.1):
                    self.speak_text(most_common_pred)
                
                return most_common_pred, confidence
        
        return self.last_prediction, self.prediction_confidence
    
    def draw_recognition_mode(self, frame, prediction, confidence, hands_detected):
        """Draw recognition mode interface"""
        height, width = frame.shape[:2]
        info_panel = np.zeros((200, width, 3), dtype=np.uint8)
        info_panel[:] = (50, 50, 50)
        
        # Mode indicator
        cv2.putText(info_panel, "MODE: RECOGNITION", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        if hands_detected and prediction and confidence > 0.3:
            cv2.putText(info_panel, f"Sign: {prediction.upper()}", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(info_panel, f"Confidence: {confidence:.2f}", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Confidence bar
            bar_width = int((width - 200) * confidence)
            cv2.rectangle(info_panel, (20, 110), (20 + bar_width, 130), (0, 255, 0), -1)
            cv2.rectangle(info_panel, (20, 110), (width - 20, 130), (255, 255, 255), 2)
        else:
            cv2.putText(info_panel, "Show hand to camera", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Speed indicator
        cv2.putText(info_panel, f"Speed: {self.recognition_speed.upper()}", 
                   (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return np.vstack([frame, info_panel])
    
    def draw_practice_mode(self, frame, prediction, confidence, hands_detected):
        """Draw practice mode interface"""
        height, width = frame.shape[:2]
        info_panel = np.zeros((250, width, 3), dtype=np.uint8)
        info_panel[:] = (40, 40, 80)
        
        # Mode indicator
        cv2.putText(info_panel, "MODE: PRACTICE", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        if self.practice_target_sign:
            cv2.putText(info_panel, f"Practice: {self.practice_target_sign.upper()}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # Show current prediction
            if hands_detected and prediction:
                success = (prediction.lower() == self.practice_target_sign.lower() and 
                          confidence > 0.6)
                color = (0, 255, 0) if success else (0, 0, 255)
                status = "CORRECT!" if success else "Try again"
                
                cv2.putText(info_panel, f"Your sign: {prediction}", 
                           (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(info_panel, status, 
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Record attempt (with some delay to avoid spam)
                current_time = time.time()
                if hasattr(self, 'last_practice_record') and current_time - self.last_practice_record > 2.0:
                    self.practice_session.record_attempt(
                        self.practice_target_sign, prediction, confidence, success
                    )
                    self.last_practice_record = current_time
                elif not hasattr(self, 'last_practice_record'):
                    self.last_practice_record = current_time
            
            # Show stats
            stats = self.practice_session.get_stats()
            cv2.putText(info_panel, f"Attempts: {stats['attempts']} | Accuracy: {stats['accuracy']:.1f}%", 
                       (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show feedback
            feedback = self.practice_session.get_feedback()
            cv2.putText(info_panel, feedback[:50], 
                       (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(info_panel, "Press number key to select sign to practice", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return np.vstack([frame, info_panel])
    
    def draw_dictionary_mode(self, frame):
        """Draw dictionary mode interface"""
        height, width = frame.shape[:2]
        info_panel = np.zeros((300, width, 3), dtype=np.uint8)
        info_panel[:] = (80, 40, 40)
        
        # Mode indicator
        cv2.putText(info_panel, "MODE: SIGN DICTIONARY", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        signs = list(self.sign_dictionary.get_all_signs())
        if signs:
            current_sign = signs[self.dictionary_current_sign % len(signs)]
            sign_info = self.sign_dictionary.get_sign_info(current_sign)
            
            cv2.putText(info_panel, f"Sign: {sign_info['name']}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            cv2.putText(info_panel, sign_info['description'], 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show first few instructions
            y_pos = 120
            for i, instruction in enumerate(sign_info['instructions'][:4]):
                cv2.putText(info_panel, f"{i+1}. {instruction[:60]}", 
                           (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_pos += 20
            
            # Navigation info
            cv2.putText(info_panel, f"Sign {self.dictionary_current_sign + 1} of {len(signs)}", 
                       (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(info_panel, "Use LEFT/RIGHT arrows to navigate", 
                       (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return np.vstack([frame, info_panel])
    
    def draw_controls_info(self, frame):
        """Draw control instructions"""
        controls = [
            "CONTROLS:",
            "'1' Recognition | '2' Practice | '3' Dictionary",
            "'r/f/n' Speed (slow/fast/normal) | 'a' Audio toggle | 's' Speak current",
            "'q' Quit | Arrow keys: Navigate dictionary"
        ]
        
        y_start = frame.shape[0] - 80
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (10, y_start + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run_enhanced_recognition(self):
        """Run the enhanced recognition system"""
        if self.model is None:
            return
        
        print("=== Enhanced Sign Language Recognition ===")
        print("Features: Practice Mode | Sign Dictionary | Speed Adjustment")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.speak_text("Enhanced sign language recognition started", force=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks and get prediction
            landmarks, hands_detected, results = self.extract_hand_landmarks(frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Make prediction
            current_prediction = ""
            current_confidence = 0.0
            
            if hands_detected and any(landmarks):
                try:
                    prediction_probs = self.model.predict_proba([landmarks])[0]
                    predicted_class_idx = np.argmax(prediction_probs)
                    predicted_class = self.model.classes_[predicted_class_idx]
                    confidence = prediction_probs[predicted_class_idx]
                    
                    current_prediction, current_confidence = self.get_smoothed_prediction(
                        predicted_class, confidence
                    )
                    
                    settings = self.speed_settings[self.recognition_speed]
                    if confidence > settings['confidence_threshold']:
                        self.last_prediction = current_prediction
                        self.prediction_confidence = current_confidence
                        self.last_detection_time = time.time()
                except Exception as e:
                    print(f"Prediction error: {e}")
            
            # Use recent prediction if available
            if time.time() - self.last_detection_time < 2.0:
                current_prediction = self.last_prediction
                current_confidence = self.prediction_confidence
            
            # Draw appropriate interface based on mode
            if self.current_mode == 'recognition':
                display_frame = self.draw_recognition_mode(
                    frame, current_prediction, current_confidence, hands_detected
                )
            elif self.current_mode == 'practice':
                display_frame = self.draw_practice_mode(
                    frame, current_prediction, current_confidence, hands_detected
                )
            elif self.current_mode == 'dictionary':
                display_frame = self.draw_dictionary_mode(frame)
            
            # Add controls info
            display_frame = self.draw_controls_info(display_frame)
            
            cv2.imshow('Enhanced Sign Language Recognition', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.current_mode = 'recognition'
                print("Switched to Recognition Mode")
            elif key == ord('2'):
                self.current_mode = 'practice'
                self.practice_session.reset_session()
                print("Switched to Practice Mode")
            elif key == ord('3'):
                self.current_mode = 'dictionary'
                print("Switched to Dictionary Mode")
            elif key == ord('a'):
                self.audio_enabled = not self.audio_enabled
                print(f"Audio {'enabled' if self.audio_enabled else 'disabled'}")
            elif key == ord('s'):
                # Manual audio repetition (original functionality)
                if current_prediction:
                    self.speak_text(current_prediction, force=True)
                    print(f"Speaking: {current_prediction}")
            elif key == ord('r'):
                self.change_speed_setting('slow')
            elif key == ord('f'):
                self.change_speed_setting('fast')
            elif key == ord('n'):
                self.change_speed_setting('normal')
            elif key == 83:  # Right arrow
                if self.current_mode == 'dictionary':
                    self.dictionary_current_sign += 1
            elif key == 81:  # Left arrow
                if self.current_mode == 'dictionary':
                    self.dictionary_current_sign = max(0, self.dictionary_current_sign - 1)
            
            # Practice mode sign selection
            if self.current_mode == 'practice':
                signs = ['hello', '3', '2', '1', 'Thank You', 'Yes', 'No']
                if key >= ord('4') and key <= ord('9'):
                    idx = key - ord('4')
                    if idx < len(signs):
                        self.practice_target_sign = signs[idx]
                        self.practice_session.reset_session()
                        print(f"Practice target set to: {self.practice_target_sign}")
        
        self.audio_queue.put("STOP")
        cap.release()
        cv2.destroyAllWindows()
        print("Enhanced recognition stopped.")

def main():
    print("=== Enhanced Sign Language Recognition System ===")
    print("NEW FEATURES:")
    print("1. Practice Mode - Learn specific signs with feedback")
    print("2. Sign Dictionary - View instructions for each sign")
    print("3. Speed Adjustment - Adjust recognition sensitivity")
    print("\nMake sure your model is trained before starting!")
    
    recognizer = EnhancedSignRecognition()
    recognizer.run_enhanced_recognition()

if __name__ == "__main__":
    main()