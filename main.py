from email.mime import image
from pyexpat import model
from turtle import color
from unittest import result
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time 
import mediapipe as mp

#--
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
# mp_holistic.POSE_CONNECTIONS
# mp_drawing.draw_landmark

def draw_styles_landmarks(image , results):
    #draw face connection
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                mp_drawing.DrawingSpec(color=(80,110,10),thickness = 1, circle_radius = 1),
                                mp_drawing.DrawingSpec(color=(80,256,121),thickness = 1, circle_radius = 1) )

    #draw pose connection
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80,22,10),thickness = 2, circle_radius = 4),
                                mp_drawing.DrawingSpec(color=(80,44,121),thickness = 2, circle_radius = 2))
    #draw hand connection
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(121,22,76),thickness = 2, circle_radius = 4),
                                mp_drawing.DrawingSpec(color=(121,44,250),thickness = 2, circle_radius = 2))

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66),thickness = 2, circle_radius = 4),
                                mp_drawing.DrawingSpec(color=(245,66,230),thickness = 2, circle_radius = 2))


cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence= 0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        #Make detection
        image, results = mediapipe_detection(frame, holistic)
        print(results.face_landmarks)
        
        #Draw landmark
        draw_styles_landmarks(image, results)
        
        cv2.imshow("Feed ", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#pose = []
#for res in results.pose_landmarks.landmark:
#    test = np.array([res.x,res.y,res.z,res.visibility])
#    print(test) 
#    pose.append(test)


def extract_key_point(results):
    pose = np.array([res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark).flatten()  if results.pose_landmarks else np.zeros(len(results.pose_landmarks.landmark)*4)
    #print(pose)

    face = np.array([res.x,res.y,res.z] for res in results.face_landmarks.landmark).flatten() if results.face_landmarks else np.zeros(len(results.face_landmarks.landmark)*3)


    lefthand = np.array([res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark).flatten() if results.left_hand_landmarks else np.zeros(len(results.left_hand_landmarks.landmark)*3)
    #print(len(results.left_hand_landmarks.landmark))

    righthand = np.array([res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark).flatten() if results.right_hand_landmarks else np.zeros(len(results.right_hand_landmarks.landmark)*3)

    return np.concatenate([pose, face, lefthand, righthand])

# lay 10 gia tri cuoi
#extract_key_point(results)[:-10]
#extract_key_point(results).shape() ra tong so landmark

# Set up folder 
# duong dan cho viec xuat du lieu
DATA_PATH = os.path.join('MP_DATA')

# hanh dong
actions = np.array(['hello', 'thks','iloveu'])

# thu thap 30 chuoi video
no_sequences = 30

# co the la chuyen video thanh 30 frame
sequences_leght = 30

for action in actions:
    for sequece in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH,action, str(sequece)))
        except:
            pass