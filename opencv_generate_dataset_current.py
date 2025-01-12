import os
import cv2
import re
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def get_highest_index(label_path, label):
    existing_files = os.listdir(label_path)
    max_index = 0
    pattern = re.compile(f"{label}_(\d+).jpg")
    for file in existing_files:
        match = pattern.match(file)
        if match:
            max_index = max(max_index, int(match.group(1)))

    return max_index


def main():
    foo = input("Index to start at: ")

    dataset_path = "data/self_curated_2/train"
    labels = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
    # remove J and Z from labels because i don't want to deal with
    # sequential motions for now
    labels.remove("J")
    labels.remove("Z")
    current_label_idx = 0 if foo == "" else labels.index(foo)
    frame_counter = 0
    capture_pressed = False
    MAX_FRAMES_TO_CAPTURE = 501
    EDGE_WARNING = 50

    os.makedirs(dataset_path, exist_ok=True)
    for label in labels:
        os.makedirs(os.path.join(dataset_path, label), exist_ok=True)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Hand Detection")
    cv2.namedWindow("Hand")

    cv2.moveWindow("Hand Detection", 0, 0)
    cv2.moveWindow("Hand", 0, 640)
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

                # find a bound box around the hand
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

                # from the bound box, find the center of the hand
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2

                # draw a bounding box centered at the center of the hand
                # bounding box must be square and have at least 50px of
                # padding between the closest landmark to a side
                side = max(x_max - x_min, y_max - y_min) // 2
                x_min = center_x - side - 25
                x_max = center_x + side + 25
                y_min = center_y - side - 25
                y_max = center_y + side + 25

                # ensure bounding box is within frame, maintaining aspect ratio
                x_min = max(0, x_min)
                x_max = min(frame.shape[1], x_max)
                y_min = max(0, y_min)
                y_max = min(frame.shape[0], y_max)

                # check if bounding box is too close to the edge
                too_close_to_edge = (
                    x_min < EDGE_WARNING
                    or x_max > frame.shape[1] - EDGE_WARNING
                    or y_min < EDGE_WARNING
                    or y_max > frame.shape[0] - EDGE_WARNING
                )

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0) if not too_close_to_edge else (0, 0, 225), 2)

                # and if we find the bounding box, crop the frame to that bounding box
                # and force resize the frame to 200x200px
                hand_frame = frame[y_min:y_max, x_min:x_max]
                if hand_frame.size > 0:
                    hand_frame = cv2.resize(hand_frame, (224, 224))
                    cv2.imshow("Hand", hand_frame)

            # write current label to bottom right of frame
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            cv2.putText(
                frame,
                f"Label: {labels[current_label_idx]}",
                (frame_width - 150, frame_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Press 'c' to capture images",
                (10, frame_height - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Captured: {frame_counter} / {MAX_FRAMES_TO_CAPTURE}",
                (10, frame_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if not capture_pressed else (0, 0, 255),
                2,
            )

            cv2.imshow("Hand Detection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c") or capture_pressed:
                capture_pressed = True
                label = labels[current_label_idx]
                label_path = os.path.join(dataset_path, label)
                os.makedirs(label_path, exist_ok=True)

                if frame_counter == 0:
                    highest_index = get_highest_index(label_path, label)
                    frame_counter = highest_index + 1 if highest_index >= 0 else 0

                cv2.imwrite(
                    os.path.join(label_path, f"{label}_{frame_counter}.jpg"), hand_frame
                )
                print(f"Captured {label}_{frame_counter}.jpg")

                frame_counter += 1

                if frame_counter - highest_index >= MAX_FRAMES_TO_CAPTURE:
                    frame_counter = 0
                    current_label_idx = (current_label_idx + 1) % len(labels)
                    capture_pressed = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
