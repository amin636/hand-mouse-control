import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
from collections import deque
import math

base_options=python.BaseOptions(model_asset_path="hand_landmarker.task")

option=vision.HandLandmarkerOptions(base_options=base_options,
                                    running_mode=vision.RunningMode.VIDEO,
                                    num_hands=1,
                                    min_hand_detection_confidence=0.6,
                                    min_hand_presence_confidence=0.5
                                    )

detector=vision.HandLandmarker.create_from_options(option)

HAND_CONNECTIONS=[
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


screen_w,screen_h=pyautogui.size()
frame_count=0

history_x=deque(maxlen=5)
history_y=deque(maxlen=5)

pyautogui.FAILSAFE = False

cliked=False
clik=45

cap=cv2.VideoCapture(0)


while(True):

    ret,frame=cap.read()

    if not ret:

        break

    frame=cv2.flip(frame,1)

    h,w,c=frame.shape

    frame_count+=1

    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=frame)

    frame_timestamp=int(frame_count*(1000/30))

    detector_result=detector.detect_for_video(mp_image,frame_timestamp)

    nok=8
    shast=4
    
    if detector_result.hand_landmarks:
        for handlandmark in detector_result.hand_landmarks:
            point=[]
           
            for landmark in handlandmark:
                cx,cy=int(landmark.x * w),int(landmark.y * h)
                point.append((cx,cy))

            x=point[nok][0]
            y=point[nok][1]

            for connect in HAND_CONNECTIONS:
                cv2.line(frame,point[connect[0]],point[connect[1]],(255,0,0),2)
                cv2.circle(frame,(x,y),5,(0,255,0),cv2.FILLED)

           
            SENSITIVITY = 2.0


            center_x, center_y = 0.5, 0.5


            dx = (x / w) - center_x
            dy = (y / h) - center_y

            x1 = int(center_x * screen_w + dx * SENSITIVITY * screen_w)
            y1 = int(center_y * screen_h + dy * SENSITIVITY * screen_h)


            x1 = max(0, min(screen_w, x1))
            y1 = max(0, min(screen_h, y1))

            history_x.append(x1)
            history_y.append(y1)

            avg_x = sum(history_x) / len(history_x)
            avg_y = sum(history_y) / len(history_y)

            pyautogui.moveTo(avg_x,avg_y)

            f=math.hypot(point[nok][0]-point[shast][0],point[nok][1]-point[shast][1])
            
            if f<clik:

                
             
                pyautogui.leftClick()
          
    cv2.imshow("h",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()