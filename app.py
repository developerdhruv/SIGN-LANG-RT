from flask import Flask, render_template, Response
import cv2
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

# Model and inference setup
index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
mean = 0.485 * 255.
std = 0.229 * 255.
ort_session = ort.InferenceSession("signlanguage.onnx")

# OpenCV video capture setup
cap = cv2.VideoCapture(0)

def center_crop(frame):
    h, w, _ = frame.shape
    start = abs(h - w) // 2
    if h > w:
        return frame[start: start + w]
    return frame[:, start: start + h]

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Preprocess frame
            frame = center_crop(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28))
            x = (resized - mean) / std
            x = x.reshape(1, 1, 28, 28).astype(np.float32)

            # Inference
            y = ort_session.run(None, {'input': x})[0]
            index = np.argmax(y, axis=1)
            letter = index_to_letter[int(index)]

            # Display result on frame
            cv2.putText(frame, letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Stream the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)