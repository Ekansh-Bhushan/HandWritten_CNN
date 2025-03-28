import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("mnist_digit_model.h5")

# Define canvas for drawing
canvas = np.zeros((400, 400), dtype=np.uint8)
drawing = False
prev_x, prev_y = None, None

def preprocess(img):
    """Preprocess the drawn image to match MNIST input"""
    img = cv2.resize(img, (28, 28))
    img = img / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)  # Reshape for the model
    return img

def predict_digit(img):
    """Predict the digit from preprocessed image"""
    processed_img = preprocess(img)
    prediction = model.predict(processed_img)
    return np.argmax(prediction)

def draw(event, x, y, flags, param):
    """Mouse callback function to draw on canvas"""
    global drawing, prev_x, prev_y
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev_x, prev_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if prev_x is not None and prev_y is not None:
            cv2.line(canvas, (prev_x, prev_y), (x, y), 255, 12)
        prev_x, prev_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        prev_x, prev_y = None, None

# Set up OpenCV window
cv2.namedWindow("Draw Digit")
cv2.setMouseCallback("Draw Digit", draw)

while True:
    img_copy = canvas.copy()
    cv2.imshow("Draw Digit", img_copy)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("p"):  # Press 'p' to predict
        digit = predict_digit(canvas)
        print("Predicted Digit:", digit)
        cv2.putText(img_copy, str(digit), (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 5)
        cv2.imshow("Draw Digit", img_copy)
        cv2.waitKey(1000)  # Pause for a second

    elif key == ord("c"):  # Press 'c' to clear canvas
        canvas.fill(0)

    elif key == ord("q"):  # Press 'q' to exit
        break

cv2.destroyAllWindows()
