import time
import cv2
import mediapipe as mp
from scamp import *

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
player = Session(tempo=100).run_as_server().new_part("organ")
last_position = {f'{i}': [0, 0, 0, ''] for i in range(4, 21, 4)}
pentatonic = [60, 62, 64, 67, 69]
for e, i in enumerate(last_position):
    last_position[f'{i}'][3] = pentatonic[e]

frame_counter = 0
average_velocity_threshold = 50  # Adjust this threshold as needed
velocity_spike_threshold = 50  # Adjust this threshold as needed

while True:
    time.sleep(0.001)
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        frame_counter += 1
        if frame_counter % 3 == 0:  # Calculate velocity every 3 frames
            finger_velocities = {}

            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id != 0 and id % 4 == 0:
                        distance = abs((abs(last_position[f'{id}'][0]**2) - (last_position[f'{id}'][1]**2)) ** 0.5 - ((abs((cx**2) + (cy**2)))) ** 0.5)
                        velocity = abs(distance - last_position[f'{id}'][2])
                        finger_velocities[id] = velocity

            max_velocity_finger = max(finger_velocities, key=finger_velocities.get)
            max_velocity = finger_velocities[max_velocity_finger]

            if max_velocity > average_velocity_threshold:
                player.play_note(last_position[f'{max_velocity_finger}'][3], 0.5, .3)

            if max_velocity_finger in last_position and abs(max_velocity - last_position[f'{max_velocity_finger}'][2]) > velocity_spike_threshold:
                player.play_note(last_position[f'{max_velocity_finger}'][3], 0.5, .3)

            frame_counter = 0

            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id != 0 and id % 4 == 0:
                    distance = abs((abs(last_position[f'{id}'][0]**2) - (last_position[f'{id}'][1]**2)) ** 0.5 - ((abs((cx**2) + (cy**2)))) ** 0.5)
                    last_position[f'{id}'] = [cx, cy, distance, last_position[f'{id}'][3]]

            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Output", image)
    cv2.waitKey(1)

