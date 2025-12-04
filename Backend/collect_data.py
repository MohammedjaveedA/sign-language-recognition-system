import cv2
import os
import numpy as np
from datetime import datetime
import mediapipe as mp
from image_preprocessing import ImagePreprocessor

class SignLanguageDataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize image preprocessor
        self.preprocessor = ImagePreprocessor()
        self.preprocessing_mode = 'enhanced'  # Default mode
        
        # Create data directory
        self.data_dir = "sign_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def create_sign_folders(self):
        """Create folders for different signs"""
        signs = ['4','3','2','1','Water','Yes','hello','Thank You','No']
        for sign in signs:
            sign_path = os.path.join(self.data_dir, sign)
            if not os.path.exists(sign_path):
                os.makedirs(sign_path)
                print(f"Created folder: {sign_path}")
    
    def extract_hand_landmarks(self, frame):
        """Extract hand landmarks from frame with preprocessing"""
        # Apply image preprocessing for better hand detection
        processed_frame = self.preprocessor.preprocess_for_hand_detection(frame, self.preprocessing_mode)
        
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        # Pad with zeros if no hands detected (63 features for 21 landmarks * 3 coordinates)
        while len(landmarks) < 63:
            landmarks.extend([0.0, 0.0, 0.0])
        
        # Limit to first hand if multiple hands detected
        return landmarks[:63], results, processed_frame
    
    def collect_data_for_sign(self, sign_name, num_samples=100):
        """Collect data for a specific sign with enhanced preprocessing"""
        print(f"\nCollecting data for '{sign_name}' sign")
        print(f"Press 's' to save image, 'q' to quit, 'n' for next sign")
        print(f"Press '1'/'2'/'3' to change preprocessing mode")
        print(f"Target: {num_samples} samples")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        sign_path = os.path.join(self.data_dir, sign_name)
        sample_count = len([f for f in os.listdir(sign_path) if f.endswith('.jpg')])
        
        landmarks_data = []
        previous_frame = None
        
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Apply frame stabilization
            if previous_frame is not None:
                frame = self.preprocessor.stabilize_frame(frame, previous_frame)
            previous_frame = frame.copy()
            
            # Extract landmarks with preprocessing
            landmarks, results, processed_frame = self.extract_hand_landmarks(frame)
            
            # Draw hand landmarks on original frame
            display_frame = frame.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Display instructions and status
            cv2.putText(display_frame, f"Sign: {sign_name.upper()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Samples: {sample_count}/{num_samples}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Mode: {self.preprocessing_mode}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, "Press 's' to save, 'q' to quit", (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "1/2/3 - Change preprocessing mode", (10, 480), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show preprocessing comparison in corner
            if processed_frame is not None:
                small_processed = cv2.resize(processed_frame, (160, 120))
                display_frame[10:130, display_frame.shape[1]-170:display_frame.shape[1]-10] = small_processed
                cv2.putText(display_frame, "Processed", (display_frame.shape[1]-165, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Enhanced Sign Language Data Collection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and results.multi_hand_landmarks:
                # Save image (original frame)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_filename = f"{sign_name}_{sample_count:03d}_{timestamp}.jpg"
                img_path = os.path.join(sign_path, img_filename)
                cv2.imwrite(img_path, frame)  # Save original frame
                
                # Extract and save landmarks
                landmarks_data.append(landmarks)
                
                sample_count += 1
                print(f"Saved sample {sample_count} for {sign_name} (Mode: {self.preprocessing_mode})")
                
            elif key == ord('q'):
                break
            elif key == ord('n'):
                break
            elif key == ord('1'):
                self.preprocessing_mode = 'standard'
                print("Switched to standard preprocessing")
            elif key == ord('2'):
                self.preprocessing_mode = 'enhanced'
                print("Switched to enhanced preprocessing")
            elif key == ord('3'):
                self.preprocessing_mode = 'robust'
                print("Switched to robust preprocessing")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save landmarks data
        if landmarks_data:
            landmarks_file = os.path.join(sign_path, f"{sign_name}_landmarks.npy")
            np.save(landmarks_file, np.array(landmarks_data))
            print(f"Saved {len(landmarks_data)} landmark samples for {sign_name}")
        
        return sample_count
    
    def run_collection(self):
        """Run the complete data collection process with preprocessing"""
        self.create_sign_folders()
        
        signs = ['4','3','2','1','Water','Yes','hello','Thank You','No']
        
        print("=== Enhanced Sign Language Data Collection ===")
        print("This tool now includes advanced image preprocessing for better accuracy.")
        print("\nFeatures:")
        print("- Enhanced contrast and brightness adjustment")
        print("- Histogram equalization for better lighting")
        print("- Skin tone enhancement for better hand detection")
        print("- Adaptive preprocessing based on lighting conditions")
        print("- Frame stabilization to reduce jitter")
        print("\nInstructions:")
        print("1. Position your hand clearly in front of the camera")
        print("2. Press 's' to save the current frame")
        print("3. Use 1/2/3 keys to change preprocessing modes:")
        print("   1 - Standard (basic enhancement)")
        print("   2 - Enhanced (recommended for most conditions)")
        print("   3 - Robust (for difficult lighting)")
        print("4. Collect at least 50-100 samples per sign for good accuracy")
        print("5. Press 'q' to quit or 'n' to move to next sign")
        
        for sign in signs:
            print(f"\n{'='*50}")
            input(f"Ready to collect data for '{sign}' sign? Press Enter to start...")
            samples_collected = self.collect_data_for_sign(sign, num_samples=100)
            print(f"Collected {samples_collected} samples for {sign}")
        
        print("\nEnhanced data collection complete!")
        print(f"Data saved in: {self.data_dir}")
        print("The preprocessing should improve model accuracy significantly!")

if __name__ == "__main__":
    collector = SignLanguageDataCollector()
    collector.run_collection()