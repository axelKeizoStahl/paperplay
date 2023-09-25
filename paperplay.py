import time
import cv2
import mediapipe as mp
from scamp import *
import threading

# Function to play sound in a separate thread
def play_sound(note, duration):
    s = Session()
    s.new_part('Jazz').play_note(note, .8, duration)

def play_chord(chord, duration):
    s = Session()
    s.new_part('Jazz').play_chord(chord, .8, duration)

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
last_position = {f'{i}': [0, 0, 0, ''] for i in range(4, 21, 4)}
pentatonic = ['47', '49', '51', '54', '56']
for e, i in enumerate(last_position):
    last_position[f'{i}'][3] = pentatonic[e]

while True:
    time.sleep(0.001)
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    if results.multi_hand_landmarks:
        for handIdx, handLms in enumerate(results.multi_hand_landmarks):
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id != 0 and id % 4 == 0:
                    distance = abs((abs(last_position[f'{id}'][0]**2)-(last_position[f'{id}'][1]**2))**0.5 - ((abs((cx**2)+(cy**2))))**0.5)
                    velocity = abs(distance - last_position[f'{id}'][2])

                    if handIdx == 0:  # First hand plays chords
                        if velocity > 30:  # Adjust this threshold as needed
                            # Play sound in a separate thread
                            #sound_thread = threading.Thread(target=play_chord, args=(last_position[f'{id}'][3], 0.5))
                            #sound_thread.start()
                            play_chord(last_position[f'{id}'][3], 0.5)
                    elif handIdx == 1:  # Second hand plays notes
                        if velocity > 30:  # Adjust this threshold as needed
                            # Play sound in a separate thread
                            #sound_thread = threading.Thread(target=play_sound, args=(last_position[f'{id}'][3], 0.5))
                            #sound_thread.start()
                            play_sound(last_position[f'{id}'][3], 0.5)

                    last_position[f'{id}'] = [cx, cy, distance, last_position[f'{id}'][3]]

            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
    cv2.imshow("Output", image)
    cv2.waitKey(1)

