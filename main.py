import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("mnist_digit_model.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Create a blank canvas for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = None, None

# Initial Drawing Parameters
draw_color = (0, 0, 255)  # Start with Red color
eraser_mode = False

def preprocess(img):
    """Preprocess the drawn image for model input."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def predict_digit(img):
    """Predict digit from the drawn image."""
    processed_img = preprocess(img)
    prediction = model.predict(processed_img)
    return np.argmax(prediction)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for natural drawing
    h, w, _ = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip landmark
            middle_finger_tip = hand_landmarks.landmark[12]  # Middle finger
            ring_finger_tip = hand_landmarks.landmark[16]  # Ring finger
            
            # Convert normalized coordinates to pixel values
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Gesture Recognition
            fingers_up = [
                hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,  # Index Finger
                hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,  # Middle Finger
                hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,  # Ring Finger
            ]

            # Mode Selection Based on Finger Count
            if fingers_up == [True, False, False]:  # One Finger (Draw Mode)
                eraser_mode = False
            elif fingers_up == [True, True, False]:  # Two Fingers (Eraser Mode)
                eraser_mode = True
            elif fingers_up == [True, True, True]:  # Three Fingers (Change Color)
                if draw_color == (0, 0, 255):  # Red → Green
                    draw_color = (0, 255, 0)
                elif draw_color == (0, 255, 0):  # Green → Blue
                    draw_color = (255, 0, 0)
                else:  # Blue → Red
                    draw_color = (0, 0, 255)

            # Draw on Canvas
            if prev_x is not None and prev_y is not None:
                if eraser_mode:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 25)  # Erase
                else:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 12)  # Draw

            prev_x, prev_y = x, y

    # Blend the video frame with the canvas
    canvas = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))  # Match sizes
    if len(canvas.shape) == 2:  # Convert grayscale to BGR
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    blended_frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)


    # Show Mode & Color
    cv2.putText(blended_frame, "Mode: Eraser" if eraser_mode else "Mode: Draw",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    
    cv2.putText(blended_frame, f"Color: {draw_color}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 3)

    cv2.imshow("Air Writing", blended_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
