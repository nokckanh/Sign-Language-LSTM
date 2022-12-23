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
    """ mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                mp_drawing.DrawingSpec(color=(80,110,10),thickness = 1, circle_radius = 1),
                                mp_drawing.DrawingSpec(color=(80,256,121),thickness = 1, circle_radius = 1) )
   """

    #draw pose connection
    """
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80,22,10),thickness = 2, circle_radius = 4),
                                mp_drawing.DrawingSpec(color=(80,44,121),thickness = 2, circle_radius = 2))
    """
    
    #draw hand connection
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(121,22,76),thickness = 2, circle_radius = 4),
                                mp_drawing.DrawingSpec(color=(121,44,250),thickness = 2, circle_radius = 2))

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66),thickness = 2, circle_radius = 4),
                                mp_drawing.DrawingSpec(color=(245,66,230),thickness = 2, circle_radius = 2))

#pose = []
#for res in results.pose_landmarks.landmark:
#    test = np.array([res.x,res.y,res.z,res.visibility])
#    print(test) 
#    pose.append(test)


def extract_key_point(results):
    #pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)

    #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)

    lefthand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    righthand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    #return np.concatenate([pose, face, lefthand, righthand])
    return np.concatenate([ lefthand, righthand])

# lay 10 gia tri cuoi
#extract_key_point(results)[:-10]
#extract_key_point(results).shape() ra tong so landmark

# Set up folder 
# duong dan cho viec xuat du lieu
DATA_PATH = os.path.join('MP_DATA')

# hanh dong
actions = np.array(['A','G'])

# thu thap 30 chuoi video
no_sequences = 30

# co the la chuyen video thanh 30 frame
sequences_leght = 30

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH,action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # NEW LOOP
    for action in actions:
        # loop thought sequences aka video
        for sequence in range(no_sequences):
            # loop thought video length aka sequence legth
            for frame_num in range(sequences_leght):

                ret, frame = cap.read()

                #Make detection
                image, results = mediapipe_detection(frame, holistic)
                #print(results.face_landmarks)
                
                #Draw landmark
                draw_styles_landmarks(image, results)

                # NEW Apply wait logic
                 # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                
                # NEW Export keypoints
                keypoints = extract_key_point(results)
                print(keypoints)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    cap.release()
    cv2.destroyAllWindows()
    
                    
cap.release()
cv2.destroyAllWindows()