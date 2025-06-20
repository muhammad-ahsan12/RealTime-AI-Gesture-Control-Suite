import cv2
import mediapipe as mp
import time
from pynput.keyboard import Controller

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Keyboard controller
keyboard = Controller()

# Control variables
last_action_time = 0
action_cooldown = 0.3  # Small cooldown to prevent rapid repeats (in seconds)
last_gesture = "none"

# Game control mappings (Poki.com version uses different keys)
JUMP_KEY = 'w'  # Poki.com uses W for jump
SLIDE_KEY = 's'  # Poki.com uses S for slide
LEFT_KEY = 'a'   # Poki.com uses A for left
RIGHT_KEY = 'd'  # Poki.com uses D for right

def get_gesture(hand_landmarks):
    """Determine gesture based on index finger position"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    wrist_x = wrist.x
    wrist_y = wrist.y
    index_x = index_tip.x
    index_y = index_tip.y
    
    # Gesture detection based on index finger position relative to wrist
    if abs(index_y - wrist_y) < 0.1 and abs(index_x - wrist_x) < 0.1:  # Near neutral
        return "none"
    elif index_y < wrist_y - 0.2:  # Index finger pointing up
        return "jump"
    elif index_y > wrist_y + 0.2:  # Index finger pointing down
        return "slide"
    elif index_x < wrist_x - 0.15:  # Index finger pointing left
        return "left"
    elif index_x > wrist_x + 0.15:  # Index finger pointing right
        return "right"
    
    return "none"

def press_key(key):
    """Press and release key with proper timing"""
    keyboard.press(key)
    time.sleep(0.1)  # Short press duration
    keyboard.release(key)

def main():
    global last_action_time, last_gesture
    
    print("Starting gesture control for Subway Surfers on Poki.com...")
    print("1. Open https://poki.com/en/g/subway-surfers in Chrome/Firefox")
    print("2. Click on the game to focus it")
    print("3. Perform index finger gestures in front of your webcam")
    print("4. Actions will trigger immediately but only once per gesture")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        # Make image writeable by copying it
        image = image.copy()
        
        # Process image
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # Reset action display
        action_text = "No action"
        action_color = (0, 0, 255)  # Red
        current_gesture = "none"
        
        # Process hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get current gesture
                current_gesture = get_gesture(hand_landmarks)
                
                # Only trigger action if:
                # 1. We have a valid gesture (not "none")
                # 2. The gesture is different from last one
                # 3. Cooldown period has passed
                if (current_gesture != "none" and 
                    current_gesture != last_gesture and 
                    time.time() - last_action_time >= action_cooldown):
                    
                    if current_gesture == "jump":
                        press_key(JUMP_KEY)
                        action_text = "JUMP (W)"
                    elif current_gesture == "slide":
                        press_key(SLIDE_KEY)
                        action_text = "SLIDE (S)"
                    elif current_gesture == "left":
                        press_key(LEFT_KEY)
                        action_text = "LEFT (A)"
                    elif current_gesture == "right":
                        press_key(RIGHT_KEY)
                        action_text = "RIGHT (D)"
                    
                    action_color = (0, 255, 0)  # Green
                    last_action_time = time.time()
                
                # Update last gesture
                last_gesture = current_gesture
                
                # Draw hand landmarks on a copy of the image
                annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Convert back to BGR for display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Display action text
        cv2.putText(image, action_text, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, action_color, 2)
        
        # Display current gesture
        cv2.putText(image, f"Gesture: {current_gesture}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # # Display instructions
        # cv2.putText(image, "Index near neutral: No action", (10, 130), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        # cv2.putText(image, "Index up: JUMP", (10, 160), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        # cv2.putText(image, "Index down: SLIDE", (10, 190), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        # cv2.putText(image, "Index left: MOVE LEFT", (10, 220), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        # cv2.putText(image, "Index right: MOVE RIGHT", (10, 250), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        # cv2.putText(image, "Press ESC to quit", (10, 280), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow('Subway Surfers Gesture Control (Poki.com)', image)
        
        # Exit on ESC key
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()