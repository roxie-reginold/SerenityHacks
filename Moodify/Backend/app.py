from flask import Flask, render_template, jsonify, Response, redirect
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import os

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'facial_expression_model_architecture.h5')
model = load_model(model_path)

emotion_log_file = 'emotion_log.txt' # Specify the path to your text file

# Global counter to keep track of the number of detected emotions
detected_emotions_count = 0

def generate_frames(detected_emotions_count):
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

        # Save detected emotion to the text file only if it's the 5th one
        if detected_emotions_count == 4:
            save_emotion_to_file(detected_emotion)
            detected_emotions_count += 1
        elif detected_emotions_count < 4:
            detected_emotions_count += 1

        cv2.putText(frame, f'Emotion: {detected_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def save_emotion_to_file(emotion):
    with open(emotion_log_file, 'a') as file:
        file.write(f'{emotion}\n')

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(detected_emotions_count), mimetype='multipart/x-mixed-replace; boundary=frame')

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

    if detected_emotion == 'Happy':
        return render_template('Happy.html', detected_emotion=detected_emotion)
    elif detected_emotion == 'Sad':
        return render_template('Sad.html', detected_emotion=detected_emotion)
    else:
        return render_template('Main.html', detected_emotion=detected_emotion)
    
@app.route('/AI.html')
def AI_html():
    return render_template('AI.html')

@app.route('/Manual.html')
def Manual_html():
    return render_template('Manual.html')

@app.route('/Happy.html')
def Happy_html():
    return render_template('Happy.html')

@app.route('/Sad.html')
def Sad_html():
    return render_template('Sad.html')

@app.route('/Frontend/Angry/options.html')
def Angry_html():
    return render_template('Angry/options.html')

@app.route('/Main.html')
def Main_html():
    return render_template('Main.html')


if __name__ == '__main__':
    app.run(debug=True)