# Hand Gesture Recognition

This project uses OpenCV and MediaPipe to perform real-time hand gesture recognition from a webcam feed. The program detects hands, identifies landmarks, and recognizes specific gestures such as "Thumbs Up", "Peace Sign", and "Fist".

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Code Overview](#code-overview)
  - [Initialization](#initialization)
  - [Gesture Recognition Function](#gesture-recognition-function)
  - [Main Loop](#main-loop)
- [License](#license)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/hand-gesture-recognition.git
   cd hand-gesture-recognition

2. **Install the required packages:**
   ```bash
    pip install opencv-python mediapipe
   
## Usage
1. Run the script:
    ```bash
    python hand_gesture_recognition.py

2. Press q to quit the application.

## Features
- Real-Time Hand Detection: Uses MediaPipe to detect hands and landmarks in real-time.
- Gesture Recognition: Recognizes specific hand gestures such as "Thumbs Up", "Peace Sign", and "Fist".
- FPS Display: Displays the current frames per second (FPS) on the video feed.

## Code Overview
Initialization
```python
import cv2
import mediapipe as mp
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Variables for calculating FPS
pTime = 0
cTime = 0
```

## Gesture Recognition Function
```python
def recognize_gesture(hand_landmarks):
    # Extracting landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    # Define gestures
    if thumb_tip.y < thumb_ip.y < index_mcp.y:
        return "Thumbs Up"
    if (index_tip.y < index_mcp.y < middle_tip.y < middle_mcp.y and
            ring_tip.y > ring_mcp.y and pinky_tip.y > pinky_mcp.y and thumb_tip.y > thumb_ip.y):
        return "Peace Sign"
    if (thumb_tip.y > thumb_ip.y and index_tip.y > index_mcp.y and
            middle_tip.y > middle_mcp.y and ring_tip.y > ring_mcp.y and pinky_tip.y > pinky_mcp.y):
        return "Fist"
    return None
```

## Main Loop
```python
while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    gesture = None
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(handLms)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    if gesture:
        cv2.putText(img, f'Gesture: {gesture}', (10, 130), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
