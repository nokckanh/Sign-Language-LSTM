
from ctypes import resize
from time import time
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
ffmpeg_extract_subclip("D:\Documents\Code Python\Đồ Án Tốt Nghiệp\Video_data.mp4", 9, 14, targetname="test.mp4")

import numpy as np
import cv2
import time
cap = cv2.VideoCapture('test.mp4')
while(cap.isOpened()):
    
    ret, frame = cap.read() 
    #cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)\

    
    
    if ret:
        
        cv2.imshow("Image", resize)
    else:
       print('no video')
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
       continue
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    time.sleep(0.02)
    

cap.release()
cv2.destroyAllWindows()
