#Modified by Augmented Startups 2021
#Face Landmark User Interface with StreamLit
#Watch Computer Vision Tutorials at www.augmentedstartups.info/YouTube
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import tensorflow.python.keras
from keras.models import load_model
import os

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

DEMO_VIDEO = 'test.mp4'
DEMO_IMAGE = 'demo.jpg'

#------------ FUNCTION--------------
def extract_key_point(results):
    lefthand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    righthand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([ lefthand, righthand])

sequence = []
sentence = []
predictions = []
threshold = 0.95
model = tensorflow.keras.models.load_model('D:\Documents\CodePython\Graduation\ModelLsmsMain.h5')
actions = np.array(['A','B','BAN','BIET','BUON','C','CAM ON','CHAP NHAN','D','DEL','E','G','GAP LAI','H','HEN','HIEU',
                            'I','K','KHOE','KHONG BIET','KHONG HIEU','L','M','N','NGAC NHIEN','NONG TINH','O','P','Q','R'
                            ,'RAT VUI DUOC GAP BAN'
                            ,'S','SO','T','TEN LA','THAT TUYET VOI','THUONG','TINH CAM','TOI','U','V','VO TAY','X','XAU HO','XIN LOI','Y'])


#------- Trainning
DATA_PATH = os.path.join('MP_DATA')

# hanh dong

# thu thap 30 chuoi video
no_sequences = 30

# co the la chuyen video thanh 30 frame
sequences_leght = 30
#------------ INTERFACE-------------

st.title('APPLICATION OF COMPUTER VISION IN COMMUNICATION')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('APPLICATION OF COMPUTER VISION IN COMMUNICATION')
st.sidebar.subheader('Parameters')

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Collection','Run on Video']
)

if app_mode =='About App':
    st.markdown('''
          # Ứng dụng
            Đây là ứng dụng phục vụ với mục đích nghiên cứu, học hỏi và giúp đỡ người khiếm thính trong giao tiếp. \n

            Ứng dụng được xây dựng dựa trên kiến trúc mạng nơ ron hồi quy và học máy. Nên việc mở rộng và phát triển là 
            hoàn toàn có thể. Ứng dụng có thể đc mở rộng và tự phát triển bởi người dùng.

            Các chức năng chính của ứng dụng chỉ là tạm thời và được phát triển tiếp trong tương lai.\n

            Các chức năng chính của ứng dụng:
            - Thu thập dữ liệu.
            - Nhận diện.
            - Hướng dẫn tập luyện cho người mới bắt đầu.
            - Truyền dữ liệu sang màn hình LCD
             
            ''')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.markdown('''
          # Hướng dẫn sử dụng  \n 

            Cấu trúc ứng dụng...
  
            ''')

    st.markdown('''
          # Về Tôi \n 
            Em tên là **Dương Văn Hiếu** hiện đang là sinh viên trường **Việt - Hàn**. \n
           
            Mục đích nghiên cứu và phát triển:
            - Học hỏi, nghiên cứu về trí tuệ nhân tạo, học máy
            - Tìm hiểu về kỹ thuật lập trình
            - Xây dựng mô hình học máy có độ chính xác cao, có khả năng ứng dụng trong thực tế.
            - Tìm kiếm giải pháp giúp đỡ những người muốn tiếp cận về ngôn ngữ kí hiệu. \n
            
            
        
            Mọi câu hỏi cần được giải đáp về ứng dụng xin liên hệ qua **Gmail: dvhieu.18it4@vku.udn.vn**
             
            ''')
elif app_mode =='Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)


    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
    
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('test.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=4)

    kpi1, kpi2, kpi3 = st.beta_columns(3)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Language**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5 ) as hand_mesh:
        prevTime = 0
        lstPridict = []
        while vid.isOpened():
            i +=1
            ret, frame = vid.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hand_mesh.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
            
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            keypoint = extract_key_point(results)
            sequence.append(keypoint)
            sequence = sequence[-30:]
            cv2.waitKey(20)
            #---- LOGIC ----
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
                                        kpi2_text.write(f"<h2 style='text-align: center; color: red;'>{actions[np.argmax(res)]}</h2>", unsafe_allow_html=True)
                                    if actions[np.argmax(res)] == 'Y':
                                        sentence.pop()
                                    lstPridict.append(actions[np.argmax(res)])   
                            except:
                                lstPridict.append(actions[np.argmax(res)]) 

                            print('hst: ',lstPridict)
                        else:
                            sentence.append(actions[np.argmax(res)])
                if len(sentence) > 5: 
                    sentence = sentence[-5:]
                    


                cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(frame, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)    

            #---------------

            
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            if record:
                #st.checkbox("Recording", value=True)
                out.write(frame)
            #Dashboard

            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            stframe.image(frame,channels = 'BGR',use_column_width=True)

    st.text('Video Processed')

    output_video = open('test.mp4','rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out. release()

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

elif app_mode =='Collection':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )

    st.markdown(' ## Đầu vào')
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    text_input = st.text_input(
        "Enter some Action 👇")
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    if text_input:
        for sequencetrain in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH,text_input, str(sequencetrain)))
            except:
                pass
        vid = cv2.VideoCapture(0)
        drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=4)
        st.markdown("<hr/>", unsafe_allow_html=True)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hand_mesh:
        # NEW LOOP
            # loop thought sequences aka video
            for sequencetrain in range(no_sequences):
                # loop thought video length aka sequence legth
                for frame_num in range(sequences_leght):

                    ret, frame = vid.read()

                    #Make detection
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hand_mesh.process(frame)

                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    #-- Draw
                    mp_drawing.draw_landmarks(
                        frame,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
                    
                    mp_drawing.draw_landmarks(
                        frame,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)

                    # NEW Apply wait logic
                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(frame, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(text_input, sequencetrain), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(text_input, sequencetrain), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                    
                    
                    # NEW Export keypoints
                    keypoints = extract_key_point(results)
                    print(keypoints)
                    npy_path = os.path.join(DATA_PATH, text_input, str(sequencetrain), str(frame_num))
                    np.save(npy_path, keypoints)
        

                #---------------
                frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
                frame = image_resize(image = frame, width = 640)
                stframe.image(frame,channels = 'BGR',use_column_width=True)
            vid.release()
