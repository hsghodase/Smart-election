import cv2
import pickle
import numpy as np
import os
import re
import time

# Configuration constants
FRAME_COUNT = 50
CAPTURE_INTERVAL = 2
FACE_RESIZE_DIMENSIONS = (500, 500)
DATA_DIRECTORY = 'data/'
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Ensure data directory exists
os.makedirs(DATA_DIRECTORY, exist_ok=True)

class AadhaarValidator:
    """Class for validating Aadhaar numbers using the Verhoeff algorithm."""
    
    def __init__(self):
        # Multiplication table
        self.d_table = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
            [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
            [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
            [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
            [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
            [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
            [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
            [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        ]
        
        # Permutation table
        self.p_table = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
            [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
            [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
            [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
            [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
            [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
            [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
        ]
        
        # Inverse table
        self.inv_table = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]
    
    def verify_checksum(self, aadhaar_number):
        """
        Validate Aadhaar number using Verhoeff algorithm.
        
        Args:
            aadhaar_number (str): The Aadhaar number to validate
            
        Returns:
            bool: True if the checksum is valid, False otherwise
        """
        c = 0
        digits = [int(x) for x in reversed(aadhaar_number)]
        
        for i, digit in enumerate(digits):
            c = self.d_table[c][self.p_table[i % 8][digit]]
        
        return c == 0
    
    def is_valid(self, aadhaar_number):
        """
        Check if the Aadhaar number is valid (12 digits and passes Verhoeff check).
        
        Args:
            aadhaar_number (str): The Aadhaar number to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check for 12 digits format and Verhoeff algorithm validation
        return bool(re.fullmatch(r'\d{12}', aadhaar_number)) and self.verify_checksum(aadhaar_number)


class FaceDataCollector:
    """Class for collecting face data from webcam."""
    
    def __init__(self, frame_count, capture_interval, face_dimensions):
        """
        Initialize the face data collector.
        
        Args:
            frame_count (int): Number of face frames to collect
            capture_interval (int): Frames to skip between captures
            face_dimensions (tuple): Width and height to resize faces to
        """
        self.frame_count = frame_count
        self.capture_interval = capture_interval
        self.face_dimensions = face_dimensions
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        
    def collect_face_data(self, identifier):
        """
        Collect face data using webcam.
        
        Args:
            identifier (str): Identifier for the collected face data
            
        Returns:
            numpy.ndarray: Collected face data
        """
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            print("Error: Could not open webcam.")
            return None
            
        faces_data = []
        frame_counter = 0
        collected_frames = 0
        
        print("\nFace data collection started. Looking for your face...")
        print("Please look at the camera and keep your face visible.")
        print(f"Collecting {self.frame_count} face samples...")
        
        while True:
            ret, frame = video.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break
                
            # Display instructions on frame
            cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Process detected faces
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Process face if it's time to capture a frame
                if frame_counter % self.capture_interval == 0 and collected_frames < self.frame_count:
                    # Crop, resize and store face
                    face_img = frame[y:y + h, x:x + w]
                    resized_face = cv2.resize(face_img, self.face_dimensions)
                    faces_data.append(resized_face)
                    collected_frames += 1
                    
                    # Show progress
                    progress = int((collected_frames / self.frame_count) * 100)
                    cv2.putText(frame, f"Progress: {progress}%", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show count of collected frames
                cv2.putText(frame, f"Collected: {collected_frames}/{self.frame_count}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                frame_counter += 1
                
            # Display the frame
            cv2.imshow('Face Registration', frame)
            
            # Check for key press or completion
            key = cv2.waitKey(1)
            if key == ord('q') or collected_frames >= self.frame_count:
                break
                
        video.release()
        cv2.destroyAllWindows()
        
        # Convert collected faces to numpy array
        if faces_data:
            faces_array = np.asarray(faces_data)
            faces_array = faces_array.reshape((len(faces_data), -1))
            print(f"Successfully collected {len(faces_data)} face images.")
            return faces_array
        else:
            print("No face data collected!")
            return None


class DataManager:
    """Class for managing face and identifier data storage."""
    
    def __init__(self, data_dir):
        """
        Initialize the data manager.
        
        Args:
            data_dir (str): Directory to store data files
        """
        self.data_dir = data_dir
        self.faces_file = os.path.join(data_dir, 'faces_data.pkl')
        self.identifiers_file = os.path.join(data_dir, 'identifiers.pkl')
    
    def save_data(self, face_data, identifier):
        """
        Save face data and identifier.
        
        Args:
            face_data (numpy.ndarray): Face data to save
            identifier (str): Identifier for the face data
        """
        # Check if files exist and load existing data
        identifiers = []
        if os.path.exists(self.identifiers_file):
            with open(self.identifiers_file, 'rb') as f:
                identifiers = pickle.load(f)
                
        faces = None
        if os.path.exists(self.faces_file):
            with open(self.faces_file, 'rb') as f:
                faces = pickle.load(f)
        
        # Add new data
        frame_count = len(face_data)
        identifiers.extend([identifier] * frame_count)
        
        if faces is None:
            faces = face_data
        else:
            faces = np.append(faces, face_data, axis=0)
        
        # Save updated data
        with open(self.identifiers_file, 'wb') as f:
            pickle.dump(identifiers, f)
            
        with open(self.faces_file, 'wb') as f:
            pickle.dump(faces, f)
            
        print(f"Data for identifier '{identifier}' saved successfully.")


def main():
    """Main function to run the Aadhaar face registration system."""
    print("=" * 60)
    print("      AADHAAR FACE REGISTRATION SYSTEM      ")
    print("=" * 60)
    
    # Validate Aadhaar number
    validator = AadhaarValidator()
    aadhaar_number = None
    
    while True:
        aadhaar_number = input("\nEnter your 12-digit Aadhaar number: ")
        if validator.is_valid(aadhaar_number):
            print("✓ Aadhaar number validated successfully.")
            break
        else:
            print("✗ Invalid Aadhaar number! Please enter a valid 12-digit Aadhaar number.")
    
    # Initialize face collector and data manager
    collector = FaceDataCollector(
        frame_count=FRAME_COUNT,
        capture_interval=CAPTURE_INTERVAL,
        face_dimensions=FACE_RESIZE_DIMENSIONS
    )
    
    data_manager = DataManager(DATA_DIRECTORY)
    
    # Collect face data
    print("\nPreparing to capture your face images...")
    time.sleep(1)
    face_data = collector.collect_face_data(aadhaar_number)
    
    if face_data is not None and len(face_data) > 0:
        # Save collected data
        data_manager.save_data(face_data, aadhaar_number)
        print("\nRegistration complete! Your face data has been linked to your Aadhaar number.")
        print(f"Total images captured and stored: {len(face_data)}")
    else:
        print("\nRegistration failed! No face data was collected.")
        print("Please try again with proper lighting and face positioning.")


if __name__ == "__main__":
    main() 