// Start live video feed
const videoFeed = document.getElementById('video-feed');
videoFeed.src = 'http://localhost:5000/video_feed';

// Optional: Add a result display
const resultDiv = document.getElementById('result');
resultDiv.innerText = 'Gesture detection is running...';