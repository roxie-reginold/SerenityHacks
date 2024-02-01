# Real-time Emotion Detection Web Application

## Overview
This is a Flask-based web application that performs real-time emotion detection using a pre-trained deep learning model. The application captures video from the user's webcam, processes each frame, and displays the detected emotion dynamically.

## Features
- Real-time video feed from the webcam.
- Emotion detection using a pre-trained deep learning model.
- Web interface to view the video feed and detected emotions.
- HTML interface

## Prerequisites
- Python 3.x
- Flask
- OpenCV
- TensorFlow (or any other deep learning library used for the emotion detection model)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/real-time-emotion-detection.git
Navigate to the project directory:

bash
Copy code
cd real-time-emotion-detection
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Run the Flask application:

bash
Copy code
python app.py
Open your web browser and go to http://localhost:5000.

Navigate to the "AI" section to view the real-time emotion detection video feed.

Configuration
You can customize the model architecture and weights files in the app.py file.
Adjust the video resolution and other parameters in the HTML and Python code as needed.

License
This project is licensed under the License - see the LICENSE file for details.

Open source used: https://github.com/nebez/floppybird
