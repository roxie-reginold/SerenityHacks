const emotionText = document.getElementById('detected-emotion');
const video = document.getElementById('video');

const updateVideo = () => {
    fetch('/video_feed')
        .then(response => response.blob())
        .then(blob => {
            const url = URL.createObjectURL(blob);
            video.src = url;
        })
        .catch(error => console.error('Error updating video:', error));
};

const updateEmotion = () => {
    fetch('/get_detected_emotion')
        .then(response => response.json())
        .then(data => {
            const detectedEmotion = data.detected_emotion;
            emotionText.textContent = detectedEmotion;
        })
        .catch(error => console.error('Error updating emotion:', error));
};

// Update the video and emotion periodically
setInterval(() => {
    updateVideo();
    updateEmotion();
}, 5000);  // Update every 1000 milliseconds (1 second)
