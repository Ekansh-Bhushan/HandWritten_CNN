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
canvas = np.zeros((480, 640), dtype=np.uint8)
drawing = False
prev_x, prev_y = None, None

def preprocess(img):
    """Preprocess the drawn image for model input."""
    img = cv2.resize(img, (28, 28))  # Resize to MNIST size
    img = img / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)  # Reshape for the model
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

    frame = cv2.flip(frame, 1)  # Flip horizontally for natural drawing
    h, w, _ = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip landmark

            # Convert normalized coordinates to pixel values
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (x, y), 255, 12)  # Draw on canvas

            prev_x, prev_y = x, y
            print("Number of landmarks detected:", len(hand_landmarks.landmark) if hand_landmarks else 0)

            finger_indices = [8, 12, 16, 20]

            if hand_landmarks and len(hand_landmarks.landmark) >= 21:
                fingers = [hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y for i in finger_indices]
            else:
                fingers = []



            if all(fingers):  # If all fingers are open (hand open)
                digit = predict_digit(canvas)
                print("Predicted Digit:", digit)
                cv2.putText(frame, f"Digit: {digit}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                cv2.imshow("Canvas", canvas)
                cv2.waitKey(1000)  # Pause for 1 second

                # Clear canvas after prediction
                canvas.fill(0)

    # Display results
    # Convert the drawing canvas to grayscale before blending
    frame_gray = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # Ensure both frames have the same shape
    if frame.shape[:2] != frame_gray.shape[:2]:  
        frame_gray = cv2.resize(frame_gray, (frame.shape[1], frame.shape[0]))

    # Blend the video frame with the drawn image
    combined = cv2.addWeighted(frame, 0.7, frame_gray, 0.3, 0)


    
    cv2.imshow("Digit Drawing", combined)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
