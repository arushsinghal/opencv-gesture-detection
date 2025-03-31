import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

def fingers_up(landmarks):
    fingers = []

    # Thumb (assumes right hand only)
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)  # Thumb up
    else:
        fingers.append(0)

    # Other four fingers
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]

    for tip, pip in zip(tips, pip_joints):
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    msg = ""

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            lm_list = handLms.landmark
            fingers = fingers_up(lm_list)

            if fingers == [1, 1, 1, 1, 1]:
                msg = "Hi"
            elif fingers == [1, 0, 0, 0, 0]:
                msg = "Thumbs Up"
            elif fingers == [0, 1, 1, 0, 0]:
                msg = "Peace"

    if msg:
        cv2.putText(frame, msg, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
