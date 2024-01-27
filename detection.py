import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('SerenityHacks/facial_expression_model_architecture.h5')  
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 for the default camera

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_resized = cv2.resize(gray, (48, 48))

    normalized_frame = gray_resized / 255.0

    input_data = np.expand_dims(np.expand_dims(normalized_frame, axis=-1), axis=0)

    # Make a prediction
    emotion_probabilities = model.predict(input_data)[0]
    predicted_emotion = emotion_labels[np.argmax(emotion_probabilities)]

    cv2.putText(frame, f'Emotion: {predicted_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Real-time Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
