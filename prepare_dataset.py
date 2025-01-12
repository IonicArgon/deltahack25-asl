import kagglehub
import os
import cv2
import mediapipe as mp
import tqdm
import logging

logging.disable()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


def impose_landmarks(image):
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                )

        return image


def process_dataset(dataset_dir, output_base_dir):
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        output_dir = os.path.join(output_base_dir, label)

        os.makedirs(output_dir, exist_ok=True)

        for image_file in tqdm.tqdm(os.listdir(label_dir)):
            image_path = os.path.join(label_dir, image_file)
            image = cv2.imread(image_path)

            if image is None:
                continue

            image = impose_landmarks(image)

            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, image)


def main():
    path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    train_dataset_dir = os.path.join(path, "asl_alphabet_train/asl_alphabet_train")
    test_dataset_dir = os.path.join(path, "asl_alphabet_test/asl_alphabet_test")

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    process_dataset(train_dataset_dir, "data/train")
    process_dataset(test_dataset_dir, "data/test")


if __name__ == "__main__":
    main()
