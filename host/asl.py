import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import json

ROWS_PER_FRAME = 543  # Number of holistic landmarks
data_columns = 3  # 'x', 'y', 'z' for each landmark
BUFFER_SIZE = 5  # Number of frames to buffer before making a prediction
landmarks_buffer = []
holistic = None
interpreter = None
index_to_label = {}

def init_asl():
    global holistic, interpreter, index_to_label

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False, 
                                    min_detection_confidence=0.5, 
                                    min_tracking_confidence=0.5)

    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    def load_label_map(json_file_path):
        with open(json_file_path, 'r') as file:
            label_map = json.load(file)
        return {v: k for k, v in label_map.items()}

    index_to_label = load_label_map("sign_to_prediction_index_map.json")

def extract_landmarks(results):
    landmarks = {'face': results.face_landmarks, 'left_hand': results.left_hand_landmarks,
                 'pose': results.pose_landmarks, 'right_hand': results.right_hand_landmarks}
    all_landmarks = []
    for key, result in landmarks.items():
        num_landmarks = {'face': 468, 'left_hand': 21, 'pose': 33, 'right_hand': 21}[key]
        if result is None:
            all_landmarks.extend([(0, 0, 0)] * num_landmarks)
        else:
            all_landmarks.extend([(landmark.x, landmark.y, landmark.z) for landmark in result.landmark])
    return all_landmarks

def update_buffer(landmarks_buffer, new_landmarks, buffer_size):
    landmarks_buffer.append(new_landmarks)
    if len(landmarks_buffer) > buffer_size:
        landmarks_buffer.pop(0)
    return landmarks_buffer

def process_asl(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    landmarks = extract_landmarks(results)
    global landmarks_buffer
    landmarks_buffer = update_buffer(landmarks_buffer, landmarks, BUFFER_SIZE)
    predicted_label = None
    confidence = None

    if len(landmarks_buffer) == BUFFER_SIZE:
        flat_list = [item for sublist in landmarks_buffer for item in sublist]
        df = pd.DataFrame(flat_list, columns=['x', 'y', 'z'])
        n_frames = int(len(df) / ROWS_PER_FRAME)
        df = df.values.reshape(n_frames, ROWS_PER_FRAME, data_columns)
        df = df.astype(np.float32)
        
        prediction_fcn = interpreter.get_signature_runner('serving_default')
        output = prediction_fcn(inputs=df)
        p = output['outputs'].reshape(-1)
        p_normalized = np.exp(p) / np.sum(np.exp(p))
        predicted_index = np.argmax(p)
        confidence = p_normalized[predicted_index]

        if confidence > 0.25:
            predicted_label = index_to_label[predicted_index]
        else:
            predicted_label = None

    return predicted_label, confidence
