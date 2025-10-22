import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp
import cv2

# Simple CNN model for landmark features
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

class SignLanguageModelTrainer:
    def __init__(self, data_dir="sign_data"):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
    
    def extract_landmarks_from_image(self, image_path):
        """Extract hand landmarks from a single image"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        # Pad with zeros if no hands detected (63 features for 21 landmarks * 3 coordinates)
        while len(landmarks) < 63:
            landmarks.extend([0.0, 0.0, 0.0])
        
        # Return first 63 features (first hand)
        return landmarks[:63]
    
    def load_data_from_images(self):
        """Load training data from saved images"""
        X = []  # Features (landmarks)
        y = []  # Labels (sign names)
        
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} not found!")
            return np.array([]), np.array([])
        
        signs = [d for d in os.listdir(self.data_dir) 
                if os.path.isdir(os.path.join(self.data_dir, d))]
        
        print(f"Found sign categories: {signs}")
        
        for sign in signs:
            sign_path = os.path.join(self.data_dir, sign)
            image_files = [f for f in os.listdir(sign_path) if f.endswith('.jpg')]
            
            print(f"Processing {len(image_files)} images for '{sign}' sign...")
            
            for img_file in image_files:
                img_path = os.path.join(sign_path, img_file)
                landmarks = self.extract_landmarks_from_image(img_path)
                
                if landmarks and any(landmarks):  # Check if landmarks were found
                    X.append(landmarks)
                    y.append(sign)
        
        print(f"Total samples loaded: {len(X)}")
        return np.array(X), np.array(y)
    
    def load_data_from_landmarks(self):
        """Load training data from saved landmark files"""
        X = []  # Features
        y = []  # Labels
        
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} not found!")
            return np.array([]), np.array([])
        
        signs = [d for d in os.listdir(self.data_dir) 
                if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for sign in signs:
            sign_path = os.path.join(self.data_dir, sign)
            landmarks_file = os.path.join(sign_path, f"{sign}_landmarks.npy")
            
            if os.path.exists(landmarks_file):
                landmarks_data = np.load(landmarks_file)
                print(f"Loaded {len(landmarks_data)} samples for '{sign}' sign")
                
                for landmarks in landmarks_data:
                    X.append(landmarks)
                    y.append(sign)
        
        return np.array(X), np.array(y)
    
    def train_model(self):
        """Train the CNN model"""
        print("Loading training data...")
        
        # Try loading from landmark files first, then from images
        X, y = self.load_data_from_landmarks()
        
        if len(X) == 0:
            print("No landmark files found. Processing images...")
            X, y = self.load_data_from_images()
        
        if len(X) == 0:
            print("No training data found! Please run collect_data.py first.")
            return False
        
        print(f"Training data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Sign classes: {np.unique(y)}")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize and train the model
        cnn_model = SignLanguageCNN(num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
        
        print("Training CNN model...")
        num_epochs = 100
        
        for epoch in range(num_epochs):
            cnn_model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = cnn_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # Create wrapper that behaves like RandomForest
        self.model = CNNWrapper(cnn_model, label_encoder, self.device)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        y_test_labels = [label_encoder.classes_[idx] for idx in y_test]
        
        accuracy = accuracy_score(y_test_labels, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred))
        
        # Save the model exactly like RandomForest
        model_path = "sign_language_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"\nModel saved as: {model_path}")
        return True
    
    def test_model_on_sample(self):
        """Test the trained model on a sample"""
        if self.model is None:
            try:
                with open("sign_language_model.pkl", 'rb') as f:
                    self.model = pickle.load(f)
            except FileNotFoundError:
                print("No trained model found. Please train the model first.")
                return
        
        # Load some test data
        X, y = self.load_data_from_landmarks()
        if len(X) == 0:
            X, y = self.load_data_from_images()
        
        if len(X) > 0:
            # Test on a random sample
            idx = np.random.randint(0, len(X))
            prediction = self.model.predict([X[idx]])
            actual = y[idx]
            
            print(f"\nSample Test:")
            print(f"Actual sign: {actual}")
            print(f"Predicted sign: {prediction[0]}")
            print(f"Correct: {'Yes' if prediction[0] == actual else 'No'}")

if __name__ == "__main__":
    trainer = SignLanguageModelTrainer()
    
    print("=== CNN Sign Language Model Trainer ===")
    print("Using CNN instead of Random Forest for better accuracy")
    
    if trainer.train_model():
        print("\nTesting model on a sample...")
        trainer.test_model_on_sample()
    else:
        print("Training failed. Please check your data.")