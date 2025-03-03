from flask import Flask, Response
from flask_cors import CORS
import cv2
import mediapipe as mp
import copy
import itertools
import csv
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Load gesture classification labels
def load_labels(file_path):
    try:
        with open(file_path, encoding='utf-8-sig') as f:
            labels = csv.reader(f)
            return [row[0] for row in labels]
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return []
    except Exception as e:
        print(f"Error loading labels: {e}")
        return []

# Load labels for keypoint classifier
keypoint_classifier_labels = load_labels('model/keypoint_classifier/keypoint_classifier_label.csv')

# KeyPointClassifier class (from the provided code)
class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index

# Initialize KeyPointClassifier
keypoint_classifier = KeyPointClassifier()

# Preprocess landmarks for classification
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
    return temp_landmark_list

# Calculate landmark list
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

# Generate video frames with gesture classification
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Process the frame with Mediapipe Hands
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw hand landmarks and classify gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Classify gesture
                landmark_list = calc_landmark_list(frame, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                gesture = keypoint_classifier_labels[hand_sign_id]

                # Display gesture label on the frame
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return "Welcome to the Live Finger Gesture Detection Backend!"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=5000)