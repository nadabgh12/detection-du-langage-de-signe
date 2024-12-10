import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = r"C:\Users\nadab\OneDrive\Bureau\Sign language detection\Data\Hello\Data\Hello"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        if y-offset >= 0 and y+h+offset <= img.shape[0] and x-offset >= 0 and x+w+offset <= img.shape[1]:
            imgCrop = img[y-offset: y+h+offset, x-offset: x+w+offset]
            imgCropShape = imgCrop.shape
            aspectratio = h / w

            if aspectratio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hGap + hCal, :] = imgResize

            cv2.imshow('Image Crop', imgCrop)
            cv2.imshow('Image White', imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{str(counter).zfill(3)}.jpg', imgWhite)
        print(f"Image saved: {counter}")
