import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1
)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Cursor smoothing
smoothing = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Click detection
click_threshold = 0.04
click_cooldown = 0.3
last_click_time = 0

# Right click detection (pinky and thumb)
right_click_threshold = 0.05
right_click_cooldown = 0.5
last_right_click_time = 0

# Drag detection
drag_threshold = 0.035
is_dragging = False

# Scroll detection
scroll_threshold = 0.04
scroll_cooldown = 0.2
last_scroll_time = 0

# Trail effect
trail_points = deque(maxlen=20)
trail_colors = deque(maxlen=20)

# UI colors
ui_color = (255, 150, 50)
highlight_color = (100, 255, 200)

# Mode selection
modes = ["Cursor", "Draw", "Scroll"]
current_mode = 0
mode_change_cooldown = 1
last_mode_change = 0

# Drawing canvas
canvas = None
draw_color = (255, 255, 255)
brush_size = 5

# Performance tracking
pTime = 0
fps_history = deque(maxlen=10)

# Reduce pyautogui delay
pyautogui.PAUSE = 0.01
pyautogui.FAILSAFE = False

def draw_ui(frame, mode, fps):
    """Draw attractive user interface elements"""
    h, w = frame.shape[:2]
    
    # Mode display
    cv2.rectangle(frame, (10, 10), (200, 50), (40, 40, 40), -1)
    cv2.putText(frame, f"Mode: {modes[mode]}", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_color, 2)
    
    # FPS display
    cv2.rectangle(frame, (w-110, 10), (w-10, 50), (40, 40, 40), -1)
    cv2.putText(frame, f"FPS: {int(fps)}", (w-100, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_color, 2)
    
    # Instructions
    cv2.putText(frame, "Pinch to click", (10, h-60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Pinky+Thumb to right click", (10, h-30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Mode switch indicator
    cv2.circle(frame, (w//2, 30), 10, highlight_color if current_mode == 0 else ui_color, 2)
    cv2.circle(frame, (w//2-30, 30), 10, highlight_color if current_mode == 1 else ui_color, 2)
    cv2.circle(frame, (w//2+30, 30), 10, highlight_color if current_mode == 2 else ui_color, 2)

def change_mode():
    """Cycle through available modes"""
    global current_mode, last_mode_change
    current_time = time.time()
    if current_time - last_mode_change > mode_change_cooldown:
        current_mode = (current_mode + 1) % len(modes)
        last_mode_change = current_time
        return True
    return False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Initialize canvas
    if canvas is None:
        canvas = np.zeros_like(frame)
    
    # Convert to RGB and process
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmarks
            landmarks = hand_landmarks.landmark
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            
            # Convert coordinates
            x, y = int(index_tip.x * w), int(index_tip.y * h)
            screen_x = np.interp(index_tip.x, [0.1, 0.9], [0, screen_w])
            screen_y = np.interp(index_tip.y, [0.1, 0.9], [0, screen_h])
            
            # Smooth cursor movement
            clocX = plocX + (screen_x - plocX) / smoothing
            clocY = plocY + (screen_y - plocY) / smoothing
            clocX = max(0, min(screen_w, clocX))
            clocY = max(0, min(screen_h, clocY))
            
            # Add point to trail
            trail_points.appendleft((x, y))
            trail_colors.appendleft((0, 255, 0))
            
            # Mode-specific functionality
            current_time = time.time()
            
            if current_mode == 0:  # Cursor mode
                pyautogui.moveTo(clocX, clocY)
                
                # Left click (index + thumb)
                distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
                if distance < click_threshold:
                    if not is_dragging and (current_time - last_click_time) > click_cooldown:
                        pyautogui.click()
                        last_click_time = current_time
                        trail_colors[0] = (0, 0, 255)  # Red for click
                    is_dragging = True
                else:
                    is_dragging = False
                
                # Right click (pinky + thumb)
                r_distance = math.hypot(thumb_tip.x - pinky_tip.x, thumb_tip.y - pinky_tip.y)
                if r_distance < right_click_threshold and (current_time - last_right_click_time) > right_click_cooldown:
                    pyautogui.rightClick()
                    last_right_click_time = current_time
                    trail_colors[0] = (255, 0, 0)  # Blue for right click
                
                # Drag functionality
                if is_dragging and distance < drag_threshold:
                    pyautogui.dragTo(clocX, clocY, button='left')
                
            elif current_mode == 1:  # Draw mode
                # Draw on canvas when pinching
                distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
                if distance < click_threshold:
                    if len(trail_points) > 1:
                        prev_x, prev_y = trail_points[1]
                        cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_size)
                    trail_colors[0] = draw_color
                
                # Change brush color with middle finger
                m_distance = math.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y)
                if m_distance < click_threshold and (current_time - last_click_time) > click_cooldown:
                    draw_color = tuple(np.random.randint(0, 255, 3).tolist())
                    brush_size = np.random.randint(3, 10)
                    last_click_time = current_time
                
            elif current_mode == 2:  # Scroll mode
                # Vertical scrolling
                distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
                if distance < scroll_threshold and (current_time - last_scroll_time) > scroll_cooldown:
                    scroll_amount = int((index_tip.y - thumb_tip.y) * 750)
                    pyautogui.scroll(scroll_amount)
                    last_scroll_time = current_time
                    trail_colors[0] = (255, 255, 0)  # Yellow for scroll
            
            # Change mode with hand wave
            mode_distance = math.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y)
            if mode_distance < 0.04 and (current_time - last_mode_change) > mode_change_cooldown:
                if change_mode():
                    trail_colors[0] = highlight_color
            
            plocX, plocY = clocX, clocY
            
            # Draw hand landmarks (semi-transparent)
            annotated_image = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1)
            )
            frame = cv2.addWeighted(frame, 0.7, annotated_image, 0.3, 0)
    
    # Draw trail effect
    for i in range(1, len(trail_points)):
        if trail_points[i-1] is None or trail_points[i] is None:
            continue
        thickness = int(np.sqrt(20 / float(i + 1)) * 2)
        cv2.line(frame, trail_points[i-1], trail_points[i], trail_colors[i], thickness)
    
    # Merge with drawing canvas in draw mode
    if current_mode == 1:
        frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    
    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    fps_history.append(fps)
    avg_fps = sum(fps_history) / len(fps_history)
    pTime = cTime
    
    # Draw UI
    draw_ui(frame, current_mode, avg_fps)
    
    # Show the frame
    cv2.imshow('Advanced Finger Control', frame)
    
    # Key controls
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('c') and current_mode == 1:  # Clear canvas
        canvas = np.zeros_like(frame)


cap.release()
cv2.destroyAllWindows()