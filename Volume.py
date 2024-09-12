import cv2
import numpy as np
import HandTracking as htm
import time
import autopy
import pyautogui

wCam, hCam = 640, 480
frameR = 100
smoothening = 7

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

previous_vol = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        if fingers[1] == 1 and fingers[2] == 0:
            length, img, lineInfo = detector.findDistance(4, 8, img)
            current_vol = np.interp(length, [15, 150], [0, 100])
            if current_vol != previous_vol:
                pyautogui.press("volup" if current_vol > previous_vol else "voldown")
            previous_vol = current_vol

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("AI Volume Controller", img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
