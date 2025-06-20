from ultralytics import YOLO
import mediapipe as mp
import cv2, numpy as np
detector = YOLO('best.pt', task='detect')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 1) YOLO detection
    yolo_results = detector.predict(source=frame, imgsz=320, conf=0.5, verbose=False)
    
    for res in yolo_results:
        boxes = res.boxes  # bounding boxes & class info
        for box in boxes:
            cls_id = int(box.cls[0])
            # Filter negative detections if needed
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            
            # 2) Pose detection on the cropped region
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_res = pose.process(crop_rgb)
            if mp_res.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    crop, mp_res.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            
            # 3) Overlay results
            label = detector.model.names[cls_id]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 5)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 7)

    cv2.imshow('Salat Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()


