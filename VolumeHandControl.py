import cv2
import time
import numpy as np
import math
import HandTrackingModule as htm
import alsaaudio
from collections import deque
from math import isclose

buffer = 8
pts = deque(maxlen=buffer)
pts_dif = deque(maxlen=buffer)

counter = 0
(dX, dY) = (0, 0)

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.6)
mixer = alsaaudio.Mixer()#alsaaudio.mixers[0])


minRadius = 50
maxRadius = 200
volume = mixer.getvolume()[0]
frames = []
image_count = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #sucv2.circle(img, (wCam // 2, hCam // 2), 150, (0, 0, 255), thickness=5)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (dX, dY) = (0, 0)
    coordinates = None
    if lmList:
        #print(lmList[2])
        #x1, y1 = lmList[4][1], lmList[4][2]
        x = [lmList[i][1] for i in [8, 12, 16, 20]]
        y = [lmList[i][2] for i in [8, 12, 16, 20]]
        x2 = int(np.mean(x))
        y2 = int(np.mean(y))
        #x2, y2 = lmList[8][1], lmList[8][2]
        coordinates = (x2, y2)
        #cx, cy = (x1+x2)//2, (y1+y2)//2
        #cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)


    pts.appendleft(coordinates)

    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        #print(pts[i])
        thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
        cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), thickness)

        if counter >= 10 and i == 1 and pts[-1] is not None:
            # compute the difference between the x and y
            dX = pts[i-1][0] - pts[i][0]
            dY = pts[i-1][1] - pts[i][1]
        #print(dX, dY)
    if dY >=45:
        volume -= 1
        mixer.setvolume(volume)
        print('Volume-', volume)

    if dY <= -40:
        volume += 1
        mixer.setvolume(volume)
        print('Volume+', volume)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)
    cv2.putText(img, f'Volume: {int(volume)}', (40, 100), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)
    cv2.imshow("Img", img)
    key = cv2.waitKey(1) & 0xFF
    if counter<10:
        counter += 1
    if key == ord("q"):
        break
    if key == ord("a"):
        image_count += 1
        frames.append(img)
        print("Adding new image:", image_count)

import imageio
with imageio.get_writer("gif.gif", mode="I") as writer:
    for idx, frame in enumerate(frames):
        print("Adding frame to GIF file: ", idx + 1)
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))