const video = document.createElement('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const processedImage = document.getElementById('processedImage');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.play();
    })
    .catch(err => {
        console.error("Error accessing webcam: ", err);
    });

const ws = new WebSocket(`ws://${window.location.host}/ws`);

ws.onopen = () => {
    console.log("WebSocket connection opened");
    setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL('image/jpeg');
            ws.send(dataURL);
        }
    }, 100);
};

ws.onmessage = (event) => {
    processedImage.src = event.data;
};

ws.onclose = () => {
    console.log("WebSocket connection closed");
};

ws.onerror = (error) => {
    console.error("WebSocket error: ", error);
};