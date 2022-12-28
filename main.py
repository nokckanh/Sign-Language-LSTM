from email.mime import image
from pyexpat import model
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
import time
from keras.models import load_model
import serial
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
    """mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                mp_drawing.DrawingSpec(color=(80,110,10),thickness = 1, circle_radius = 1),
                                mp_drawing.DrawingSpec(color=(80,256,121),thickness = 1, circle_radius = 1) )
"""
    #draw pose connection
    """
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80,22,10),thickness = 2, circle_radius = 4),
                                mp_drawing.DrawingSpec(color=(80,44,121),thickness = 2, circle_radius = 2))zz
    """
    
    #draw hand connection
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(121,22,76),thickness = 2, circle_radius = 4),
                                mp_drawing.DrawingSpec(color=(121,44,250),thickness = 2, circle_radius = 2))

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66),thickness = 2, circle_radius = 4),
                                mp_drawing.DrawingSpec(color=(245,66,230),thickness = 2, circle_radius = 2))


def extract_key_point(results):
  # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)

  #  face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)

    lefthand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    righthand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([ lefthand, righthand])

colors = [(245,117,16), (117,245,16), (16,117,245),(16,886,245),(16,117,225),(36,117,245),(16,54,245),(16,54,125),
            (16,2,125),(16,54,321),(16,32,125),(16,54,242),(16,54,232),(16,64,232),(16,54,45),(16,54,21),(16,54,234)
            ,(16,54,122),(16,54,621),(245,117,16),(245,117,16),(245,117,16),(245,117,16),(245,117,16),(245,117,16),
            (245,117,16),(245,117,16),(245,117,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16)
            , (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16)
            , (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16)
            , (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16)
            , (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16), (117,245,16)]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        if num > 15:
            if num == 17 :
                cv2.rectangle(output_frame, (200,60+0*40), (int(200+ prob*100), 90+0*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+0*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 18 :
                cv2.rectangle(output_frame, (200,60+1*40), (int(200+ prob*100), 90+1*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+1*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 19 :
                cv2.rectangle(output_frame, (200,60+2*40), (int(200+ prob*100), 90+2*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+2*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 20 :
                cv2.rectangle(output_frame, (200,60+3*40), (int(200+ prob*100), 90+3*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+3*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 21 :
                cv2.rectangle(output_frame, (200,60+4*40), (int(200+ prob*100), 90+4*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+4*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 22 :
                cv2.rectangle(output_frame, (200,60+5*40), (int(200+ prob*100), 90+5*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+5*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 23 :
                cv2.rectangle(output_frame, (200,60+6*40), (int(200+ prob*100), 90+6*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+6*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 24 :
                cv2.rectangle(output_frame, (200,60+7*40), (int(200+ prob*100), 90+7*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+7*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 25 :
                cv2.rectangle(output_frame, (200,60+8*40), (int(200+ prob*100), 90+8*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+8*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 26 :
                cv2.rectangle(output_frame, (200,60+9*40), (int(200+ prob*100), 90+9*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+9*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 27 :
                cv2.rectangle(output_frame, (200,60+10*40), (int(200+ prob*100), 90+10*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+10*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 28 :
                cv2.rectangle(output_frame, (200,60+11*40), (int(200+ prob*100), 90+11*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+11*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 29 :
                cv2.rectangle(output_frame, (200,60+12*40), (int(200+ prob*100), 90+12*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+12*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 30 :
                cv2.rectangle(output_frame, (200,60+13*40), (int(200+ prob*100), 90+13*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+13*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 31 : 
                cv2.rectangle(output_frame, (200,60+14*40), (int(200+ prob*100), 90+14*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+14*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 32 :
                cv2.rectangle(output_frame, (200,60+15*40), (int(200+ prob*100), 90+15*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (200, 85+15*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA) 
        if num > 32:
            if num == 33 :
                cv2.rectangle(output_frame, (410,60+0*40), (int(410+ prob*100), 90+0*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (410, 85+0*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 34 :
                cv2.rectangle(output_frame, (410,60+1*40), (int(410+ prob*100), 90+1*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (410, 85+1*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 35 :
                cv2.rectangle(output_frame, (410,60+2*40), (int(410+ prob*100), 90+2*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (410, 85+2*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 36 :
                cv2.rectangle(output_frame, (410,60+3*40), (int(410+ prob*100), 90+3*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (410, 85+3*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 37 :
                cv2.rectangle(output_frame, (410,60+4*40), (int(410+ prob*100), 90+4*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (410, 85+4*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 38 :
                cv2.rectangle(output_frame, (410,60+5*40), (int(410+ prob*100), 90+5*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (410, 85+5*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 39 :
                cv2.rectangle(output_frame, (410,60+6*40), (int(410+ prob*100), 90+6*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (410, 85+6*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 40 :
                cv2.rectangle(output_frame, (410,60+7*40), (int(410+ prob*100), 90+7*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (410, 85+7*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 41 :
                cv2.rectangle(output_frame, (410,60+8*40), (int(410+ prob*100), 90+8*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (410, 85+8*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 42 :
                cv2.rectangle(output_frame, (410,60+9*40), (int(410+ prob*100), 90+9*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (410, 85+9*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 43 :
                cv2.rectangle(output_frame, (410,60+10*40), (int(410+ prob*100), 90+10*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (410, 85+10*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 44 :
                cv2.rectangle(output_frame, (410,60+11*40), (int(410+ prob*100), 90+11*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (410, 85+11*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 45 :
                cv2.rectangle(output_frame, (410,60+12*40), (int(410+ prob*100), 90+12*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (410, 85+12*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if num == 46 :
                cv2.rectangle(output_frame, (410,60+13*40), (int(410+ prob*100), 90+13*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (410, 85+13*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                      
        else:
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def listToString(s):
    # initialize an empty string
    str1 = " "
    # return string 
    return (str1.join(s))

s = serial.Serial('com3',9600) #port is 11 (for COM12, and baud rate is 9600
  #wait for the Serial to initialize

sequence = []
sentence = []
predictions = []
threshold = 0.75
model = load_model("D:\Documents\CodePython\Graduation\ModelLsmsMain.h5")
actions = np.array(['A','B','BAN','BIET','BUON','C','CAM ON','CHAP NHAN','D','DEL','E','G','GAP LAI','H','HEN','HIEU',
                        'I','K','KHOE','KHONG BIET','KHONG HIEU','L','M','N','NGAC NHIEN','NONG TINH','O','P','Q','R'
                        ,'RAT VUI DUOC GAP BAN'
                        ,'S','SO','T','TEN LA','THAT TUYET VOI','THUONG','TINH CAM','TOI','U','V','VO TAY','X','XAU HO','XIN LOI','Y'])

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence= 0.5) as holistic:
    lstPridict = []
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
        cv2.waitKey(20)

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            #print(res)
            print(actions[np.argmax(res)])

            s = actions[np.argmax(res)]
            predictions.append(np.argmax(res))
            #-> lay ra vi tri so nao to nhat
            #-- logic viz
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        try:
                            if(actions[np.argmax(res)]!= lstPridict[len(lstPridict)-1]):
                                if actions[np.argmax(res)] != sentence[-1] and actions[np.argmax(res)] != 'Y':
                                    sentence.append(actions[np.argmax(res)])
                                if actions[np.argmax(res)] == 'Y':
                                    sentence.pop()
                                lstPridict.append(actions[np.argmax(res)])   
                        except:
                            lstPridict.append(actions[np.argmax(res)]) 

                        print('hst: ',lstPridict)
                    else:
                        sentence.append(actions[np.argmax(res)])
            if len(sentence) > 5: 
                sentence = sentence[-4:]
                s.write(listToString(sentence).encode())
                #listToStr = ' '.join(map(str, s))
            else:
                pass

            # Viz probabilities
            frame2 = np.zeros((750 , 700 , 3) , np.uint8) * 255

            frame2 = prob_viz(res, actions, frame2, colors)

            cv2.imshow("Frame2" , frame2)
            
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