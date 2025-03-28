import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the trained MNIST model
model = tf.keras.models.load_model("mnist_digit_model.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Create a blank canvas for drawing (3-channel BGR)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Initial Drawing Parameters
draw_color = (0, 0, 255)  # Start with Red color
eraser_mode = False
prev_x, prev_y = None, None

def preprocess(img):
    """Preprocess the drawn image for model input."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (28, 28))  # Resize to match MNIST input
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, 28, 28, 1)  # Reshape for model
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
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Detect raised fingers
            fingers_up = [
                hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,  # Index Finger
                hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,  # Middle Finger
                hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,  # Ring Finger
                hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y,  # Pinky
                hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y,  # Thumb
            ]

            # Mode Selection Based on Finger Count
            if fingers_up[:2] == [True, False]:  # 1 Finger (Draw Mode)
                eraser_mode = False
            elif fingers_up[:2] == [True, True]:  # 2 Fingers (Eraser Mode)
                eraser_mode = True
            elif fingers_up[:3] == [True, True, True]:  # 3 Fingers (Change Color)
                if draw_color == (0, 0, 255):  # Red → Green
                    draw_color = (0, 255, 0)
                elif draw_color == (0, 255, 0):  # Green → Blue
                    draw_color = (255, 0, 0)
                else:  # Blue → Red
                    draw_color = (0, 0, 255)

            # Draw or Erase
            if prev_x is not None and prev_y is not None:
                if eraser_mode:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 25)  # Erase
                else:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 12)  # Draw

            prev_x, prev_y = x, y

            # Predict Digit When All 5 Fingers Are Raised
            if all(fingers_up):
                digit = predict_digit(canvas)
                print("Predicted Digit:", digit)
                cv2.putText(frame, f"Digit: {digit}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                cv2.imshow("Canvas", canvas)
                cv2.waitKey(1000)  # Pause for 1 second
                canvas.fill(0)  # Clear canvas after prediction

    # Resize `canvas` to match `frame` size
    canvas = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))
    
    # Convert grayscale canvas to BGR if needed
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # Blend the video frame with the canvas
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
