import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('facial_expression_model_architecture.h5')  # Replace 'your_model.h5' with the path to your saved model

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or change to the appropriate camera index

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to match the input size of your trained model
    gray_resized = cv2.resize(gray, (48, 48))

    # Normalize the pixel values to be in the range [0, 1]
    normalized_frame = gray_resized / 255.0

    # Expand dimensions to match the model's expected input shape
    input_data = np.expand_dims(np.expand_dims(normalized_frame, axis=-1), axis=0)

    # Make a prediction
    emotion_probabilities = model.predict(input_data)[0]
    predicted_emotion = emotion_labels[np.argmax(emotion_probabilities)]

    # Display the frame with the predicted emotion
    cv2.putText(frame, f'Emotion: {predicted_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Real-time Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
