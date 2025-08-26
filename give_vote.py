from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
import threading

# Event to track when speech is completed
speech_completed = threading.Event()

def speak(str1):
    # Reset the speech completion event
    speech_completed.clear()
    
    # Run speech in a separate thread to avoid blocking the main UI
    threading.Thread(target=_speak_thread, args=(str1,), daemon=True).start()

def _speak_thread(str1):
    try:
        speech = Dispatch("SAPI.SpVoice")
        
        # Get available voices
        voices = speech.GetVoices()
        
        # Find a female voice
        female_voice = None
        
        for i in range(voices.Count):
            voice = voices.Item(i)
            voice_description = voice.GetDescription()
            if any(female_term in voice_description.lower() for female_term in ["female", "woman", "girl", "zira", "hazel"]):
                female_voice = voice
                break
        
        # Set the female voice if found
        if female_voice:
            speech.Voice = female_voice
        
        # Adjust voice properties
        speech.Rate = 0
        speech.Volume = 100
        
        # Speak the message
        speech.Speak(str1)
        
        # Signal that speech is completed
        speech_completed.set()
    except Exception as e:
        print(f"Speech error: {e}")
        # In case of error, make sure to set the event to avoid hanging
        speech_completed.set()

# Initialize camera in a separate thread for smoother startup
def initialize_camera():
    cap = cv2.VideoCapture(0)
    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

print("Initializing system...")
# Start camera initialization
video = initialize_camera()

print("Loading face detection model...")
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.exists('data/'):
    os.makedirs('data/')

try:
    # Changed from names.pkl to identifiers.pkl
    with open('data/identifiers.pkl', 'rb') as f:
        LABELS = pickle.load(f)

    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    FACES = np.array(FACES)
    LABELS = np.array(LABELS)

    min_samples = min(len(FACES), len(LABELS))
    FACES = FACES[:min_samples]
    LABELS = LABELS[:min_samples]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)
    print("Face recognition model loaded successfully")
except Exception as e:
    print(f"Error loading face data: {e}")
    exit(1)

try:
    background_img = cv2.imread("background.png")
    if background_img is None:
        background_img = np.zeros((600, 800, 3), dtype=np.uint8)
        background_img[:] = (50, 50, 50)
except:
    background_img = np.zeros((600, 800, 3), dtype=np.uint8)
    background_img[:] = (50, 50, 50)

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
background_img = cv2.resize(background_img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

# Updated camera panel to left-bottom
CAM_WIDTH = 320
CAM_HEIGHT = 240
cam_x = 50
cam_y = 200

status_text_y = cam_y + CAM_HEIGHT + 20

COL_NAMES = ['NAME', 'VOTE', 'DATE', 'TIME']

# Create CSV file explicitly at program start
csv_file_path = os.path.join(os.getcwd(), 'Votes.csv')
if not os.path.exists(csv_file_path):
    try:
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(COL_NAMES)
        print(f"Created new CSV file at: {csv_file_path}")
    except Exception as e:
        print(f"Error creating CSV file: {e}")
else:
    print(f"Using existing CSV file at: {csv_file_path}")

def check_if_exists(value):
    try:
        with open(csv_file_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == value:
                    return True
    except FileNotFoundError:
        print(f"CSV file not found at: {csv_file_path}")
    except Exception as e:
        print(f"Error checking CSV file: {e}")
    return False

# Function to record vote with voice confirmation
def record_vote(voter_name, vote_choice):
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Record the vote with explicit error handling
    try:
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([voter_name, vote_choice, current_date, current_time])
        print(f"Vote successfully recorded for {voter_name}: {vote_choice}")
        print(f"Vote saved to: {csv_file_path}")
        
        # Voice confirmation
        speak(f"Thank you {voter_name}. You have completed voting successfully.")
        return True
    except Exception as e:
        print(f"Error recording vote: {e}")
        speak(f"Error recording vote. Please try again.")
        return False

cv2.putText(background_img, "FACIAL RECOGNITION VOTING SYSTEM", 
            (DISPLAY_WIDTH//2 - 230, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

party_info_y = 120
party_info_x = DISPLAY_WIDTH - 300
cv2.putText(background_img, "VOTING OPTIONS:", 
            (party_info_x, party_info_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
cv2.putText(background_img, "1 - BJP", 
            (party_info_x, party_info_y + 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
cv2.putText(background_img, "2 - CONGRESS", 
            (party_info_x, party_info_y + 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
cv2.putText(background_img, "3 - AAP", 
            (party_info_x, party_info_y + 150), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
cv2.putText(background_img, "4 - NOTA", 
            (party_info_x, party_info_y + 200), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
cv2.putText(background_img, "ESC - Exit", 
            (party_info_x, party_info_y + 250), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

cv2.putText(background_img, "RECOGNITION PANEL", 
            (cam_x + 10, cam_y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

cv2.rectangle(background_img, 
              (cam_x - 10, cam_y - 20), 
              (cam_x + CAM_WIDTH + 10, cam_y + CAM_HEIGHT + 80), 
              (100, 100, 100), 1)

# Variables for tracking voter state
announced_voters = set()  # Track voters who have already received announcements
current_voter = None
voting_complete = False
exit_timer_started = False
exit_start_time = 0
auto_exit_delay = 5 # Exit after 3 seconds

print("System initialized. Starting main loop...")

# Create window early for better experience
cv2.namedWindow('Voting System', cv2.WINDOW_NORMAL)

while True:
    try:
        ret, frame = video.read()
        if not ret:
            print("Error reading from camera. Retrying...")
            time.sleep(0.5)
            continue

        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        output = None
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)

        display_img = background_img.copy()

        cv2.rectangle(display_img, (cam_x-2, cam_y-2), (cam_x+CAM_WIDTH+2, cam_y+CAM_HEIGHT+2), (0, 0, 255), 2)
        display_img[cam_y:cam_y + CAM_HEIGHT, cam_x:cam_x + CAM_WIDTH] = frame

        cv2.putText(display_img, "CAMERA FEED", 
                    (cam_x + CAM_WIDTH//2 - 60, cam_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Check if speech has completed and then start the exit timer
        if speech_completed.is_set() and voting_complete and not exit_timer_started:
            exit_timer_started = True
            exit_start_time = time.time()
            print("Speech announcement completed. Starting exit timer.")

        # Handle auto-exit after speech announcement is complete
        if exit_timer_started:
            remaining_time = 3 - (time.time() - exit_start_time)
            if remaining_time <= 0:
                print("Auto-exit timer complete. Exiting system.")
                break
            
            # Display countdown
            cv2.putText(display_img, f"Exiting in {int(remaining_time)}s", 
                    (DISPLAY_WIDTH//2 - 100, DISPLAY_HEIGHT - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
        # Get keyboard input
        k = cv2.waitKey(1)
        if k != -1:  # -1 means no key was pressed
            print(f"Key pressed: {k}")

        if output is not None:
            voter_name = output[0]
            print(f"Detected voter: {voter_name}")
            
            cv2.rectangle(display_img, 
                         (cam_x + 10, status_text_y - 5), 
                         (cam_x + CAM_WIDTH - 10, status_text_y + 65), 
                         (50, 50, 50), -1)

            cv2.putText(display_img, "VOTER DETECTED:", 
                        (cam_x + 20, status_text_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.rectangle(display_img, 
                         (cam_x + 20, status_text_y + 25), 
                         (cam_x + CAM_WIDTH - 20, status_text_y + 55), 
                         (0, 100, 0), -1)

            cv2.putText(display_img, voter_name, 
                        (cam_x + 30, status_text_y + 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            voter_exist = check_if_exists(voter_name)
            print(f"Has voter already voted? {voter_exist}")
            
            # Check if we have a new voter
            if current_voter != voter_name:
                current_voter = voter_name
                voting_complete = False
                exit_timer_started = False
                speech_completed.clear()  # Reset speech completion event for new voter
                print(f"New voter detected: {voter_name}")
            
            if voter_exist:
                cv2.putText(display_img, "STATUS: ALREADY VOTED", 
                            (cam_x + 20, status_text_y + 85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Announce that voter has already voted (only once per session)
                if voter_name not in announced_voters:
                    speak(f"Voter {voter_name} has already cast their vote.")
                    announced_voters.add(voter_name)  # Mark this voter as announced
                    voting_complete = True
                    print(f"Announcing that voter {voter_name} has already voted")
            else:
                cv2.putText(display_img, "STATUS: READY TO VOTE", 
                            (cam_x + 20, status_text_y + 85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Debug: Print voting eligibility
                print(f"Voter {voter_name} eligible to vote: {not voting_complete}")
                
                # Check for vote keys being pressed
                if not voting_complete:
                    if k == 49:  # 1 key for BJP
                        print(f"BJP vote key pressed for {voter_name}")
                        if record_vote(voter_name, "BJP"):
                            voting_complete = True
                            announced_voters.add(voter_name)
                    elif k == 50:  # 2 key for CONGRESS
                        print(f"CONGRESS vote key pressed for {voter_name}")
                        if record_vote(voter_name, "CONGRESS"):
                            voting_complete = True
                            announced_voters.add(voter_name)
                    elif k == 51:  # 3 key for AAP
                        print(f"AAP vote key pressed for {voter_name}")
                        if record_vote(voter_name, "AAP"):
                            voting_complete = True
                            announced_voters.add(voter_name)
                    elif k == 52:  # 4 key for NOTA
                        print(f"NOTA vote key pressed for {voter_name}")
                        if record_vote(voter_name, "NOTA"):
                            voting_complete = True
                            announced_voters.add(voter_name)
        else:
            cv2.rectangle(display_img, 
                         (cam_x + 10, status_text_y - 5), 
                         (cam_x + CAM_WIDTH - 10, status_text_y + 30), 
                         (50, 50, 50), -1)

            cv2.putText(display_img, "Waiting for face detection...", 
                        (cam_x + 20, status_text_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            # Reset current voter when no face is detected
            current_voter = None
            voting_complete = False
            exit_timer_started = False

        if not exit_timer_started:
            cv2.putText(display_img, "Press the corresponding key to cast your vote", 
                        (DISPLAY_WIDTH//2 - 200, DISPLAY_HEIGHT - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Voting System', display_img)

        if k == 27:  # ESC key
            print("ESC key pressed. Exiting.")
            break
    
    except Exception as e:
        print(f"Error in main loop: {e}")
        time.sleep(0.1)

print("Cleaning up resources...")
video.release()
cv2.destroyAllWindows()
print("System shutdown complete.")