import cv2
import numpy as np
from typing import Tuple, Optional

class ImagePreprocessor:
    """
    Image preprocessing class to enhance hand detection and landmark extraction
    """
    
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )
        self.kernel = np.ones((3, 3), np.uint8)
        
    def enhance_contrast_and_brightness(self, image: np.ndarray, alpha: float = 1.2, beta: int = 10) -> np.ndarray:
        """
        Enhance image contrast and brightness for better hand visibility
        alpha: contrast control (1.0-3.0)
        beta: brightness control (0-100)
        """
        if image is None or image.size == 0:
            return image
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced
    
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise
        """
        if image is None or image.size == 0:
            return image
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to improve contrast
        """
        if image is None or image.size == 0:
            return image
            
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except:
            # If histogram equalization fails, return original image
            return image
    
    def remove_background(self, image: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """
        Remove background using background subtraction (optional - use carefully)
        """
        fg_mask = self.background_subtractor.apply(image, learningRate=learning_rate)
        
        # Clean up the mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Apply mask to original image
        result = cv2.bitwise_and(image, image, mask=fg_mask)
        
        return result
    
    def enhance_skin_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance skin regions to improve hand detection
        """
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Clean up the mask
        skin_mask = cv2.medianBlur(skin_mask, 5)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self.kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Apply mask to enhance skin regions
        enhanced = image.copy()
        enhanced[skin_mask > 0] = cv2.convertScaleAbs(enhanced[skin_mask > 0], alpha=1.3, beta=20)
        
        return enhanced
    
    def stabilize_frame(self, current_frame: np.ndarray, previous_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply temporal smoothing to reduce frame jitter
        """
        if previous_frame is None:
            return current_frame
        
        # Simple temporal smoothing
        alpha = 0.7  # Current frame weight
        stabilized = cv2.addWeighted(current_frame, alpha, previous_frame, 1 - alpha, 0)
        
        return stabilized
    
    def preprocess_for_hand_detection(self, image: np.ndarray, mode: str = 'standard') -> np.ndarray:
        """
        Main preprocessing function with different modes
        
        Modes:
        - 'standard': Basic enhancement
        - 'enhanced': Advanced preprocessing with skin detection
        - 'robust': Maximum preprocessing for difficult conditions
        """
        
        if mode == 'standard':
            # Basic preprocessing
            processed = self.enhance_contrast_and_brightness(image, alpha=1.1, beta=5)
            processed = self.apply_gaussian_blur(processed, (3, 3))
            
        elif mode == 'enhanced':
            # Enhanced preprocessing
            processed = self.enhance_contrast_and_brightness(image, alpha=1.2, beta=10)
            processed = self.apply_histogram_equalization(processed)
            processed = self.enhance_skin_detection(processed)
            processed = self.apply_gaussian_blur(processed, (3, 3))
            
        elif mode == 'robust':
            # Maximum preprocessing for difficult lighting
            processed = self.apply_histogram_equalization(image)
            processed = self.enhance_contrast_and_brightness(processed, alpha=1.3, beta=15)
            processed = self.enhance_skin_detection(processed)
            processed = self.apply_gaussian_blur(processed, (5, 5))
            
        else:
            processed = image
        
        return processed
    
    def adaptive_preprocessing(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Automatically choose preprocessing based on image characteristics
        """
        # Calculate image statistics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Choose preprocessing mode based on image characteristics
        if mean_brightness < 80:  # Dark image
            mode = 'robust'
        elif std_brightness < 30:  # Low contrast
            mode = 'enhanced'
        else:  # Normal conditions
            mode = 'standard'
        
        processed = self.preprocess_for_hand_detection(image, mode)
        
        return processed, mode
    
    def create_hand_roi_mask(self, image: np.ndarray, hand_landmarks) -> np.ndarray:
        """
        Create a region of interest mask around detected hand
        """
        if hand_landmarks is None:
            return image
        
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get hand bounding box with padding
        x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
        
        x_min, x_max = max(0, min(x_coords) - 50), min(w, max(x_coords) + 50)
        y_min, y_max = max(0, min(y_coords) - 50), min(h, max(y_coords) + 50)
        
        # Create ROI mask
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
        
        # Apply mask
        result = cv2.bitwise_and(image, image, mask=mask)
        
        return result
    
    def get_preprocessing_info(self, mode: str) -> str:
        """
        Get information about the current preprocessing mode
        """
        info = {
            'standard': 'Basic: Contrast + Blur',
            'enhanced': 'Enhanced: Histogram + Skin',
            'robust': 'Robust: Maximum Enhancement'
        }
        return info.get(mode, 'Unknown')

# Example usage and testing
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = ImagePreprocessor()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()
    
    print("Image Preprocessing Test")
    print("Controls:")
    print("'1' - Standard preprocessing")
    print("'2' - Enhanced preprocessing") 
    print("'3' - Robust preprocessing")
    print("'a' - Adaptive preprocessing")
    print("'n' - No preprocessing")
    print("'q' - Quit")
    
    current_mode = 'standard'
    previous_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Apply preprocessing based on current mode
        if current_mode == 'none':
            processed_frame = frame
        elif current_mode == 'adaptive':
            processed_frame, detected_mode = preprocessor.adaptive_preprocessing(frame)
            cv2.putText(processed_frame, f"Auto Mode: {detected_mode}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            processed_frame = preprocessor.preprocess_for_hand_detection(frame, current_mode)
        
        # Add stabilization
        if previous_frame is not None:
            processed_frame = preprocessor.stabilize_frame(processed_frame, previous_frame)
        
        previous_frame = processed_frame.copy()
        
        # Display information
        mode_info = preprocessor.get_preprocessing_info(current_mode)
        cv2.putText(processed_frame, f"Mode: {mode_info}", 
                   (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(processed_frame, "Press 1/2/3/a/n to change mode, q to quit", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show both original and processed
        combined = np.hstack([frame, processed_frame])
        cv2.imshow('Original vs Preprocessed', combined)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_mode = 'standard'
            print("Switched to standard preprocessing")
        elif key == ord('2'):
            current_mode = 'enhanced'
            print("Switched to enhanced preprocessing")
        elif key == ord('3'):
            current_mode = 'robust'
            print("Switched to robust preprocessing")
        elif key == ord('a'):
            current_mode = 'adaptive'
            print("Switched to adaptive preprocessing")
        elif key == ord('n'):
            current_mode = 'none'
            print("Preprocessing disabled")
    
    cap.release()
    cv2.destroyAllWindows()