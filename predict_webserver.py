import cv2
import mediapipe as mp
import torch
import numpy as np
import os
from transformers import ViTFeatureExtractor, ViTForImageClassification
from flask import Flask, request, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

label2id = {
    "F": 0,
    "E": 1,
    "D": 2,
    "H": 3,
    "L": 4,
    "M": 5,
    "S": 6,
    "K": 7,
    "O": 8,
    "V": 9,
    "X": 10,
    "N": 11,
    "Q": 12,
    "G": 13,
    "T": 14,
    "U": 15,
    "B": 16,
    "Y": 17,
    "A": 18,
    "W": 19,
    "R": 20,
    "P": 21,
    "C": 22,
    "I": 23,
}

id2label = {
    0: "F",
    1: "E",
    2: "D",
    3: "H",
    4: "L",
    5: "M",
    6: "S",
    7: "K",
    8: "O",
    9: "V",
    10: "X",
    11: "N",
    12: "Q",
    13: "G",
    14: "T",
    15: "U",
    16: "B",
    17: "Y",
    18: "A",
    19: "W",
    20: "R",
    21: "P",
    22: "C",
    23: "I",
}

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained(
    "./results/ds2/checkpoint-4050",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        try:
            print(request)
            file = request.files["file"]

            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400
            
            if not allowed_file(file.filename):
                return jsonify({"error": "Invalid file type"}), 400
            
            filename = secure_filename(file.filename)
            file.save(f"{app.config['UPLOAD_FOLDER']}/dl_{filename}")

            saved_file = open(f"{app.config['UPLOAD_FOLDER']}/dl_{filename}", "rb")

            image = cv2.imdecode(np.frombuffer(saved_file.read(), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            with mp_hands.Hands(
                static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
            ) as hands:
                results = hands.process(image)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=hand_landmarks,
                            connections=mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                        )

                    hand_landmarks = results.multi_hand_landmarks[0]

                    x_min, x_max = 1, 0
                    y_min, y_max = 1, 0

                    for landmark in hand_landmarks.landmark:
                        x, y = landmark.x, landmark.y
                        x_min = min(x_min, x)
                        x_max = max(x_max, x)
                        y_min = min(y_min, y)
                        y_max = max(y_max, y)

                    x_min, x_max = int(x_min * image.shape[1]), int(x_max * image.shape[1])
                    y_min, y_max = int(y_min * image.shape[0]), int(y_max * image.shape[0])

                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2

                    side = max(x_max - x_min, y_max - y_min) // 2
                    x_min = center_x - side - 50
                    x_max = center_x + side + 50
                    y_min = center_y - side - 50
                    y_max = center_y + side + 50

                    hand_image = image[y_min:y_max, x_min:x_max]

                    if hand_image.size > 0:
                        hand_image = cv2.resize(hand_image, (224, 224))
                        hand_image = Image.fromarray(hand_image)
                        inputs = feature_extractor(images=hand_image, return_tensors="pt")

                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits
                            predicted_class_idx = logits.argmax().item()
                            predicted_label = id2label[predicted_class_idx]

                            return jsonify({"label": predicted_label})
                    else:
                        return jsonify({"error": "Hand not found"}), 400

        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
