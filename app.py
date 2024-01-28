from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Response
import os

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'facial_expression_model_architecture.h5')
model = load_model(model_path)

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion_model = model  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (48, 48))
        normalized_frame = gray_resized / 255.0
        input_data = np.expand_dims(np.expand_dims(normalized_frame, axis=-1), axis=0)

        emotion_probabilities = emotion_model.predict(input_data)[0]
        detected_emotion = emotion_labels[np.argmax(emotion_probabilities)]

        cv2.putText(frame, f'Emotion: {detected_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
app = Flask(__name__)
app.static_folder = 'static'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detected_emotion')
def get_detected_emotion():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (48, 48))
    normalized_frame = gray_resized / 255.0
    input_data = np.expand_dims(np.expand_dims(normalized_frame, axis=-1), axis=0)

    emotion_probabilities = model.predict(input_data)[0]
    detected_emotion = emotion_labels[np.argmax(emotion_probabilities)]

    return jsonify({'detected_emotion': detected_emotion})

@app.route('/')
def home():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (48, 48))
    normalized_frame = gray_resized / 255.0
    input_data = np.expand_dims(np.expand_dims(normalized_frame, axis=-1), axis=0)

    emotion_probabilities = model.predict(input_data)[0]
    detected_emotion = emotion_labels[np.argmax(emotion_probabilities)]

    return render_template('home.html', detected_emotion=detected_emotion)


if __name__ == '__main__':
    app.run(debug=True)