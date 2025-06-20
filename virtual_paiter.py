import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime

class VirtualPainter:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Canvas setup
        self.canvas = None
        self.canvas_history = []  # For undo functionality
        self.max_history = 20
        self.prev_x, self.prev_y = 0, 0
        
        # Enhanced color palette
        self.colors = [
            (255, 0, 0),      # Blue
            (0, 255, 0),      # Green
            (0, 0, 255),      # Red
            (0, 255, 255),    # Yellow
            (255, 0, 255),    # Magenta
            (255, 255, 0),    # Cyan
            (128, 0, 128),    # Purple
            (255, 165, 0),    # Orange
            (255, 192, 203),  # Pink
            (128, 128, 128),  # Gray
            (255, 255, 255),  # White
            (0, 0, 0)         # Black (eraser)
        ]
        self.current_color_index = 0
        self.current_color = self.colors[0]
        
        # Brush settings
        self.brush_sizes = [2, 5, 10, 15, 20, 30]
        self.current_brush_size_index = 1
        self.current_brush_size = self.brush_sizes[1]
        
        # Brush styles
        self.brush_styles = ['normal', 'circle', 'square', 'spray']
        self.current_brush_style_index = 0
        self.current_brush_style = self.brush_styles[0]
        
        # UI settings
        self.ui_height = 120
        self.color_palette_height = 40
        self.button_height = 35
        self.button_width = 80
        
        # Tool modes
        self.modes = ['draw', 'erase', 'text']
        self.current_mode_index = 0
        self.current_mode = self.modes[0]
        
        # Text settings
        self.font_sizes = [0.5, 0.7, 1.0, 1.5, 2.0]
        self.current_font_size_index = 2
        self.current_font_size = self.font_sizes[2]
        
        # FPS tracking
        self.prev_time = time.time()
        
        # Pinch detection
        self.pinch_threshold = 0.05
        self.is_drawing = False
        
        # Create saves directory
        if not os.path.exists('paintings'):
            os.makedirs('paintings')
    
    def save_canvas_state(self):
        """Save current canvas state for undo functionality"""
        if len(self.canvas_history) >= self.max_history:
            self.canvas_history.pop(0)
        self.canvas_history.append(self.canvas.copy())
    
    def undo_last_action(self):
        """Undo last drawing action"""
        if len(self.canvas_history) > 0:
            self.canvas = self.canvas_history.pop()
    
    def draw_ui(self, image):
        """Draw the enhanced UI"""
        h, w = image.shape[:2]
        
        # Draw UI background
        cv2.rectangle(image, (0, 0), (w, self.ui_height), (50, 50, 50), -1)
        
        # Draw color palette
        color_width = w // len(self.colors)
        for i, color in enumerate(self.colors):
            x1 = i * color_width
            x2 = (i + 1) * color_width
            cv2.rectangle(image, (x1, 10), (x2, 10 + self.color_palette_height), color, -1)
            
            # Highlight selected color
            if i == self.current_color_index:
                cv2.rectangle(image, (x1, 10), (x2, 10 + self.color_palette_height), (255, 255, 255), 3)
        
        # Draw brush size indicators
        y_start = 60
        for i, size in enumerate(self.brush_sizes):
            x = 10 + i * 60
            color = (255, 255, 255) if i == self.current_brush_size_index else (150, 150, 150)
            cv2.circle(image, (x + 20, y_start + 15), size//2, color, -1)
            cv2.putText(image, str(size), (x + 10, y_start + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw mode buttons
        mode_x_start = 400
        for i, mode in enumerate(self.modes):
            x = mode_x_start + i * 90
            color = (0, 255, 0) if i == self.current_mode_index else (100, 100, 100)
            cv2.rectangle(image, (x, y_start), (x + 80, y_start + 30), color, -1)
            cv2.putText(image, mode.upper(), (x + 10, y_start + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw function buttons
        button_x_start = 700
        buttons = ['CLEAR', 'UNDO', 'SAVE']
        for i, button in enumerate(buttons):
            x = button_x_start + i * 90
            cv2.rectangle(image, (x, y_start), (x + 80, y_start + 30), (0, 100, 200), -1)
            cv2.putText(image, button, (x + 10, y_start + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw current settings info
        info_y = y_start + 45
        cv2.putText(image, f"Brush: {self.current_brush_style} | Size: {self.current_brush_size}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Mode: {self.current_mode.upper()}", 
                   (300, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def handle_ui_click(self, x, y, w):
        """Handle clicks on UI elements"""
        if y < 10 + self.color_palette_height and y > 10:
            # Color selection
            color_width = w // len(self.colors)
            self.current_color_index = min(x // color_width, len(self.colors) - 1)
            self.current_color = self.colors[self.current_color_index]
            
        elif y >= 60 and y <= 90:
            # Brush size selection
            if x < 360:  # Brush size area
                brush_index = x // 60
                if brush_index < len(self.brush_sizes):
                    self.current_brush_size_index = brush_index
                    self.current_brush_size = self.brush_sizes[brush_index]
            
            # Mode selection
            elif x >= 400 and x < 670:
                mode_index = (x - 400) // 90
                if mode_index < len(self.modes):
                    self.current_mode_index = mode_index
                    self.current_mode = self.modes[mode_index]
            
            # Function buttons
            elif x >= 700:
                button_index = (x - 700) // 90
                if button_index == 0:  # Clear
                    self.save_canvas_state()
                    self.canvas = np.zeros_like(self.canvas)
                elif button_index == 1:  # Undo
                    self.undo_last_action()
                elif button_index == 2:  # Save
                    self.save_painting()
    
    def apply_brush_style(self, canvas, x, y, prev_x, prev_y):
        """Apply different brush styles"""
        if self.current_brush_style == 'normal':
            if prev_x != 0 and prev_y != 0:
                cv2.line(canvas, (prev_x, prev_y), (x, y), self.current_color, self.current_brush_size)
            else:
                cv2.circle(canvas, (x, y), self.current_brush_size//2, self.current_color, -1)
                
        elif self.current_brush_style == 'circle':
            cv2.circle(canvas, (x, y), self.current_brush_size//2, self.current_color, -1)
            
        elif self.current_brush_style == 'square':
            half_size = self.current_brush_size // 2
            cv2.rectangle(canvas, (x-half_size, y-half_size), 
                         (x+half_size, y+half_size), self.current_color, -1)
            
        elif self.current_brush_style == 'spray':
            for _ in range(20):
                spray_x = x + np.random.randint(-self.current_brush_size, self.current_brush_size)
                spray_y = y + np.random.randint(-self.current_brush_size, self.current_brush_size)
                cv2.circle(canvas, (spray_x, spray_y), 1, self.current_color, -1)
    
    def save_painting(self):
        """Save the current painting"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"paintings/painting_{timestamp}.png"
        
        # Create a clean version without UI
        clean_canvas = self.canvas.copy()
        cv2.imwrite(filename, clean_canvas)
        print(f"Painting saved as {filename}")
    
    def run(self):
        """Main application loop"""
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Initialize canvas on first run
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)
                self.save_canvas_state()
            
            # Convert BGR to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)
            
            # Convert back to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Get frame dimensions
            h, w = image.shape[:2]
            
            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get finger tip coordinates
                    index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                    
                    x, y = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Check if interacting with UI
                    if y < self.ui_height:
                        # Check for pinch gesture in UI area
                        distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
                        if distance < self.pinch_threshold and not self.is_drawing:
                            self.handle_ui_click(x, y, w)
                            self.is_drawing = True
                        elif distance >= self.pinch_threshold:
                            self.is_drawing = False
                        continue
                    
                    # Drawing area
                    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
                    
                    if distance < self.pinch_threshold:  # Pinch detected
                        if not self.is_drawing:
                            self.save_canvas_state()
                            self.is_drawing = True
                            self.prev_x, self.prev_y = x, y
                        
                        # Apply drawing based on current mode
                        if self.current_mode == 'draw':
                            self.apply_brush_style(self.canvas, x, y, self.prev_x, self.prev_y)
                        elif self.current_mode == 'erase':
                            cv2.circle(self.canvas, (x, y), self.current_brush_size, (0, 0, 0), -1)
                        
                        self.prev_x, self.prev_y = x, y
                    else:
                        self.is_drawing = False
                        self.prev_x, self.prev_y = 0, 0
                    
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=4)
                    )
                    
                    # Draw cursor
                    cursor_color = (0, 255, 0) if self.is_drawing else (255, 0, 0)
                    cv2.circle(image, (x, y), 10, cursor_color, 2)
            
            # Merge canvas with camera feed
            image = cv2.addWeighted(image, 0.6, self.canvas, 0.4, 0)
            
            # Draw UI
            self.draw_ui(image)
            
            # Display FPS
            cur_time = time.time()
            fps = 1 / (cur_time - self.prev_time) if (cur_time - self.prev_time) > 0 else 0
            self.prev_time = cur_time
            cv2.putText(image, f"FPS: {int(fps)}", (w-100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(image, "Pinch to draw/select | Press 'q' to quit | Press 'b' to change brush style", 
                       (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the result
            cv2.imshow('Enhanced Virtual Painter', image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.save_canvas_state()
                self.canvas = np.zeros_like(frame)
            elif key == ord('b'):
                self.current_brush_style_index = (self.current_brush_style_index + 1) % len(self.brush_styles)
                self.current_brush_style = self.brush_styles[self.current_brush_style_index]
            elif key == ord('s'):
                self.save_painting()
            elif key == ord('u'):
                self.undo_last_action()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

# Run the application
if __name__ == "__main__":
    painter = VirtualPainter()
    painter.run()