import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0) #0 is the id number of the webcam
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox'] #getting the bounding box info from hand, asking the dictionary to give us all the values

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 #matrix of ones, keeping the image square, colored image, type of values - 8 bit value = from 0 tp 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset] #starting height:ending height, starting width:ending width -- because its a matrix

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio>1: #if height is greater than width
            k = imgSize/h #k is the constant
            wCal = math.ceil(k*w) #always round up, no decimal values
            imgResize = cv2.resize(imgCrop,(wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize #height is 300 so can leave it blank

        else: #if height is greater than width
            k = imgSize/w #k is the constant
            hCal = math.ceil(k*h) #always round up, no decimal values
            imgResize = cv2.resize(imgCrop,(imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize #height is 300 so can leave it blank


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)    
    
    cv2.imshow("Image", img)
    cv2.waitKey(1) #1ms delay


    