import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ------------------- MediaPipe Setup -------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ------------------- Webcam Setup -------------------
cap = cv2.VideoCapture(0)

# ------------------- Audio Setup (Master Volume) -------------------
# This works in PyCaw 20181226
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
vol_range = volume.GetVolumeRange()
vol_min, vol_max = vol_range[0], vol_range[1]

# ------------------- Hand Detection -------------------
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # ------------------- Thumb & Index Tip -------------------
                h, w, _ = frame.shape
                x1 = int(hand_landmarks.landmark[4].x * w)  # Thumb tip
                y1 = int(hand_landmarks.landmark[4].y * h)
                x2 = int(hand_landmarks.landmark[8].x * w)  # Index tip
                y2 = int(hand_landmarks.landmark[8].y * h)

                # Draw circles & line
                cv2.circle(frame, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # ------------------- Distance & Volume Mapping -------------------
                length = math.hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, [20, 200], [vol_min, vol_max])
                volume.SetMasterVolumeLevel(vol, None)

                # Display volume percentage
                vol_percentage = int(np.interp(length, [20, 200], [0, 100]))
                cv2.putText(frame, f'Vol: {vol_percentage}%', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ------------------- Show Frame -------------------
        cv2.imshow("Hand Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

# ------------------- Release -------------------
cap.release()
cv2.destroyAllWindows()
