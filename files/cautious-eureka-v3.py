import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()
print(f"Screen size: {screen_width}x{screen_height}")

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Camera calibration parameters (may need adjustment)
# These values are estimates - you might need to calibrate for your specific camera
FOCAL_LENGTH = 600  # Approximate focal length in pixels
KNOWN_WIDTH = 8.5   # Average palm width in cm (used for calibration)

# Control parameters
pinch_threshold_cm = 5.0  # 5 cm threshold for pinch/drag transition
drag_threshold = 0.2      # Time threshold to start dragging (seconds)
cursor_smoothing = 0.5    # Smoothing factor for cursor movement

# State tracking
was_pinched = False
pinch_start_time = 0
is_dragging = False
prev_x, prev_y = 0, 0
last_click_time = 0
palm_width_pixels = 0
distance_cm = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Only track one hand for simplicity
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    
    print("Starting hand tracking. Press 'q' to quit...")
    
    while webcam.isOpened():
        success, img = webcam.read()
        if not success:
            print("Failed to read frame from webcam")
            break
        
        # Flip image horizontally for more intuitive control
        img = cv2.flip(img, 1)
        img_height, img_width, _ = img.shape
        
        # Process hand tracking
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        current_pinched = False
        hand_detected = False
        
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                # Get landmarks for thumb (4) and index finger (8)
                thumb = hand_landmarks.landmark[4]
                index_finger = hand_landmarks.landmark[8]
                
                # Convert to pixel coordinates
                thumb_x, thumb_y = int(thumb.x * img_width), int(thumb.y * img_height)
                index_x, index_y = int(index_finger.x * img_width), int(index_finger.y * img_height)
                
                # Draw connection between thumb and index finger
                cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
                cv2.circle(img, (thumb_x, thumb_y), 8, (0, 255, 255), -1)
                cv2.circle(img, (index_x, index_y), 8, (0, 255, 255), -1)
                
                # Calculate Euclidean distance in pixels
                distance_pixels = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                
                # Estimate physical distance in cm
                # 1. First calculate distance to hand using palm width
                # Get wrist (0) and index finger base (5) for width estimation
                wrist = hand_landmarks.landmark[0]
                index_base = hand_landmarks.landmark[5]
                
                # Calculate palm width in pixels
                palm_width_pixels = math.sqrt(
                    (wrist.x * img_width - index_base.x * img_width)**2 +
                    (wrist.y * img_height - index_base.y * img_height)**2
                )
                
                # 2. Estimate distance to hand in cm
                if palm_width_pixels > 0:
                    distance_to_hand = (KNOWN_WIDTH * FOCAL_LENGTH) / palm_width_pixels
                    
                    # 3. Estimate actual distance between fingers in cm
                    distance_cm = (distance_pixels * KNOWN_WIDTH) / palm_width_pixels
                
                # Check pinch condition based on physical distance
                if distance_cm < pinch_threshold_cm:
                    current_pinched = True
                    cv2.putText(img, f"PINCHING: {distance_cm:.1f}cm", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(img, f"Distance: {distance_cm:.1f}cm", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Cursor position mapping
                cursor_x = (index_x / img_width) * screen_width
                cursor_y = (index_y / img_height) * screen_height
                
                # Apply smoothing
                smoothed_x = prev_x * (1 - cursor_smoothing) + cursor_x * cursor_smoothing
                smoothed_y = prev_y * (1 - cursor_smoothing) + cursor_y * cursor_smoothing
                
                # Update cursor position
                pyautogui.moveTo(smoothed_x, smoothed_y)
                prev_x, prev_y = smoothed_x, smoothed_y
        
        # Drag/Pinch state machine based on physical distance
        if current_pinched:
            # Start dragging if we're pinching and under the distance threshold
            if not is_dragging:
                pyautogui.mouseDown()
                is_dragging = True
                cv2.putText(img, "DRAGGING", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # If we were dragging but stopped pinching, release
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
            
            # Handle clicks on pinch release
            if was_pinched:
                current_time = time.time()
                if current_time - last_click_time < 0.3:  # Double click within 300ms
                    pyautogui.doubleClick()
                    cv2.putText(img, "DOUBLE CLICK", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    pyautogui.click()
                    cv2.putText(img, "SINGLE CLICK", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                last_click_time = current_time
        
        # Update pinch state
        was_pinched = current_pinched
        
        # If hand disappears while dragging, release mouse
        if not hand_detected and is_dragging:
            pyautogui.mouseUp()
            is_dragging = False
            was_pinched = False
        
        # Display status information
        mode_text = "DRAGGING" if is_dragging else "NORMAL"
        color = (0, 255, 0) if is_dragging else (255, 255, 255)
        cv2.putText(img, f"Mode: {mode_text}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        cv2.putText(img, f"Palm width: {palm_width_pixels:.1f}px", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(img, "Press 'q' to quit", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display image
        cv2.imshow('Hand Controlled Mouse', img)
        
        # Check for quit key
        key = cv2.waitKey(5)
        if key == ord('q') or key == 27:  # 27 is ESC key
            print("Exiting...")
            break

# Cleanup
webcam.release()
cv2.destroyAllWindows()
print("Program exited successfully")
