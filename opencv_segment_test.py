import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def main():
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
                # for hand_landmarks in results.multi_hand_landmarks:
                #     mp_drawing.draw_landmarks(
                #         image=frame,
                #         landmark_list=hand_landmarks,
                #         connections=mp_hands.HAND_CONNECTIONS,
                #         landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                #         connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                #     )
                
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
                x_min = center_x - side - 50
                x_max = center_x + side + 50
                y_min = center_y - side - 50
                y_max = center_y + side + 50

                # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # and if we find the bounding box, crop the frame to that bounding box
                # and force resize the frame to 200x200px
                hand_frame = frame[y_min:y_max, x_min:x_max]
                if hand_frame.size > 0:
                    hand_frame = cv2.resize(hand_frame, (200, 200))
                    # cv2.imshow("Hand", hand_frame)


            cv2.imshow("Hand Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()