import cv2
import mediapipe as mp
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image

# Initialize MediaPipe Hands
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

# Load the model and feature extractor
model = ViTForImageClassification.from_pretrained(
    "./results/ds2/checkpoint-4050",
    num_labels=24,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Set the model to evaluation mode
model.eval()

# Initialize video capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                )

            hand_landmarks = results.multi_hand_landmarks[0]

            # Find a bounding box around the hand
            x_min, x_max = 1, 0
            y_min, y_max = 1, 0

            for landmark in hand_landmarks.landmark:
                x, y = landmark.x, landmark.y
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

            x_min, x_max = int(x_min * frame.shape[1]), int(x_max * frame.shape[1])
            y_min, y_max = int(y_min * frame.shape[0]), int(y_max * frame.shape[0])

            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            side = max(x_max - x_min, y_max - y_min) // 2
            x_min = center_x - side - 25
            x_max = center_x + side + 25
            y_min = center_y - side - 25
            y_max = center_y + side + 25

            # Crop the frame to the bounding box and resize to 224x224
            hand_frame = frame[y_min:y_max, x_min:x_max]
            if hand_frame.size > 0:
                hand_frame = cv2.resize(hand_frame, (224, 224))

                # Convert the frame to a PIL image
                hand_image = Image.fromarray(
                    cv2.cvtColor(hand_frame, cv2.COLOR_BGR2RGB)
                )

                # Preprocess the image
                inputs = feature_extractor(images=hand_image, return_tensors="pt")

                # Perform the classification
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()
                    predicted_label = id2label[predicted_class_idx]

                # Display the predicted label on the frame
                cv2.putText(
                    frame,
                    predicted_label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.imshow("Hand", hand_frame)

        cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
