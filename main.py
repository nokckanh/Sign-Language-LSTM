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

from keras.models import load_model

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


def extract_key_point(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)

    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)

    lefthand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    righthand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lefthand, righthand])

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

sequence = []
sentence = []
predictions = []
threshold = 0.8
model = load_model("D:\Documents\Code Python\Đồ Án Tốt Nghiệp\sign.h5")
actions = np.array(['hello', 'thks','iloveu'])

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence= 0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        #Make detection
        image, results = mediapipe_detection(frame, holistic)
        
        #Draw landmark
        draw_styles_landmarks(image, results)

        #predict
        keypoint = extract_key_point(results)

        sequence.append(keypoint)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            #-- logic viz
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)            

        #
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


"""
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


"""