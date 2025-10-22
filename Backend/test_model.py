#!/usr/bin/env python3
"""
Test script to verify the model is working correctly
Run this script to debug model loading and prediction issues
"""

import pickle
import numpy as np
import os
import sys

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("=== Model Loading Test ===")
    
    model_path = "sign_language_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file '{model_path}' not found!")
        print("Please ensure you have:")
        print("1. Collected training data using collect_data.py")
        print("2. Trained the model using train_model.py")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Check if model has classes_ attribute
        if hasattr(model, 'classes_'):
            print(f"Available classes: {list(model.classes_)}")
            print(f"Number of classes: {len(model.classes_)}")
        else:
            print("❌ WARNING: Model doesn't have 'classes_' attribute")
            
        return model
        
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return None

def test_model_prediction(model):
    """Test if the model can make predictions"""
    print("\n=== Model Prediction Test ===")
    
    if model is None:
        print("❌ Cannot test prediction - model not loaded")
        return False
    
    # Create dummy landmark data (63 features for 21 landmarks * 3 coordinates)
    dummy_landmarks = [0.5] * 63  # Use 0.5 instead of 0.0 for more realistic test
    
    try:
        # Test predict method
        prediction = model.predict([dummy_landmarks])
        print(f"✅ Prediction successful: {prediction[0]}")
        
        # Test predict_proba method
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([dummy_landmarks])[0]
            print(f"✅ Prediction probabilities:")
            for i, (cls, prob) in enumerate(zip(model.classes_, probabilities)):
                print(f"  {cls}: {prob:.4f} ({prob*100:.1f}%)")
        else:
            print("❌ WARNING: Model doesn't have 'predict_proba' method")
            
        return True
        
    except Exception as e:
        print(f"❌ ERROR making prediction: {e}")
        return False

def test_mediapipe():
    """Test if MediaPipe is working"""
    print("\n=== MediaPipe Test ===")
    
    try:
        import mediapipe as mp
        import cv2
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        print("✅ MediaPipe imported successfully")
        print("✅ Hands solution initialized")
        
        # Create a dummy image and test processing
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb_image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        print("✅ Image processing test successful")
        print(f"Hands detected: {results.multi_hand_landmarks is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR with MediaPipe: {e}")
        return False

def test_preprocessing():
    """Test if image preprocessing is working"""
    print("\n=== Image Preprocessing Test ===")
    
    try:
        from image_preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor()
        print("✅ ImagePreprocessor imported successfully")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test different preprocessing modes
        modes = ['standard', 'enhanced', 'robust']
        for mode in modes:
            try:
                processed = preprocessor.preprocess_for_hand_detection(dummy_image, mode)
                print(f"✅ Preprocessing mode '{mode}' working")
            except Exception as e:
                print(f"❌ ERROR with preprocessing mode '{mode}': {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR with preprocessing: {e}")
        print("Make sure image_preprocessing.py is in the same directory")
        return False

def test_data_directory():
    """Test if training data exists"""
    print("\n=== Training Data Test ===")
    
    data_dir = "sign_data"
    
    if not os.path.exists(data_dir):
        print(f"❌ Training data directory '{data_dir}' not found")
        return False
    
    signs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not signs:
        print("❌ No sign directories found in training data")
        return False
    
    print(f"✅ Found {len(signs)} sign categories:")
    
    total_samples = 0
    for sign in signs:
        sign_path = os.path.join(data_dir, sign)
        
        # Count images
        images = [f for f in os.listdir(sign_path) if f.endswith('.jpg')]
        
        # Check for landmarks file
        landmarks_file = os.path.join(sign_path, f"{sign}_landmarks.npy")
        has_landmarks = os.path.exists(landmarks_file)
        
        print(f"  {sign}: {len(images)} images, landmarks: {'✅' if has_landmarks else '❌'}")
        total_samples += len(images)
    
    print(f"Total training samples: {total_samples}")
    
    if total_samples < 50:
        print("⚠️  WARNING: Very few training samples. Collect more data for better accuracy.")
    
    return len(signs) > 0

def main():
    """Run all tests"""
    print("🧪 Sign Language Recognition - Diagnostic Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Data directory
    if test_data_directory():
        tests_passed += 1
    
    # Test 2: Model loading
    model = test_model_loading()
    if model is not None:
        tests_passed += 1
    
    # Test 3: Model prediction
    if test_model_prediction(model):
        tests_passed += 1
    
    # Test 4: MediaPipe
    if test_mediapipe():
        tests_passed += 1
    
    # Test 5: Preprocessing
    if test_preprocessing():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🧪 Test Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Your setup should work correctly.")
    elif tests_passed >= 3:
        print("⚠️  Most tests passed. Check the failed tests above.")
    else:
        print("❌ Many tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("1. Run 'python collect_data.py' to collect training data")
        print("2. Run 'python train_model.py' to train the model")
        print("3. Install missing dependencies: pip install -r requirements.txt")
        print("4. Make sure all original files are in the backend directory")

if __name__ == "__main__":
    main()