import cv2
#import zmq
import numpy as np
import dlib
#specialized tools to be included in github/gitlab lib
#from .helpers import FACIAL_LANDMARK_IDXS
#from .helpers import shape_to_np

import json
#import requests
#import socket
#from threading import *

'''
serversocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host="127.0.0.0"
port=6969
serversocket.bind((host,port))


class client(Thread):
    def __init__(self,socket,address):
        self.sock=socket
        self.addr=address
        self.start()

    def run(self):
        while True:
            #print("Client sent:",self.sock.recv(1024).decode())
            self.sock.send()
'''


class contourDetector:
    def __init__(self,predictor,desiredLeftEye=(0.35,0.35),desiredFaceWidth=256,desiredFaceHeight=None):
        self.predictor=predictor
        self.desiredLeftEye=desiredLeftEye
        self.desiredFaceWidth=desiredFaceWidth
        self.desiredFaceHeight=desiredFaceHeight

        if self.desiredFaceHeight is None:
            self.desiredFaceHeight=self.desiredFaceWidth

    def align(self,image,gray,rect):
        shape=self.predictor(gray,rect)
        shape=shape_to_np(shape)

        #extract left and right eye (x,y) coordinates
        (lStart,lEnd)=FACIAL_LANDMARK_IDXS["left_eye"]
        (rStart,rEnd)=FACIAL_LANDMARK_IDXS["right_eye"]
        leftEyePts=shape[lStart:lEnd]
        rightEyePts=shape[rStart:rEnd]

        #center of mass for each eye
        leftEyeCenter=leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter=rightEyePts.mean(axis=0).astype("int")

        #compute angle between eye centroids
        dY=rightEyeCenter[1]-leftEyeCenter[1]
        dX=rightEyeCenter[0]-leftEyeCenter[0]
        angle=np.degrees(np.arctan2(dY,dX))-100

        #compute desired right eye X coordinate based on X-coordinate of left eye
        desiredRightEyeX=1.0-self.desiredLeftEye[0]

        #determine scale of image by taking ratio pf distance between eyes in current image to ratio of distance
        #between eyes in the desired image
        dist=np.sqrt((dX**2)+(dY**2))
        desiredDist=(desiredRightEyeX-self.desiredLeftEye[0])
        desiredDist*=self.desiredFaceWidth
        scale=desiredDist/dist

        #compute center (x,y)-coordinates between two eyes in input image
        eyesCenter=((leftEyeCenter[0]+rightEyeCenter[0])//2,(leftEyeCenter[1]+rightEyeCenter[1])//2)

        #get rotation matrix for rotating and scaling the face
        M=cv2.getRotationMatrix2D(eyesCenter,angle,scale)

        #update translation component of the matrix
        tX=self.desiredFaceWidth*0.5
        tY=self.desiredFaceHeight*self.desiredLeftEye[1]
        M[0,2]+=(tX-eyesCenter[0])
        M[1,2]+=(tY-eyesCenter[1])

        #appliy affine transformation
        (w,h)=(self.desiredFaceWidth,self.desiredFaceHeight)
        output=cv2.wrapAffine(image,M,(w,h),flags=cv2.INTER_CUBIC)

        #return aligned face
        return output

def rect_to_bb(rect):
    #take bounding box predicted by dlib and convert it to (x,y,w,h) {used by OpenCV}
    x=rect.left()
    y=rect.top()
    w=rect.right()-x
    h=rect.bottom()-y
    #return a tuple of (x,y,w,h)
    return (x,y,w,h)

#TEST/RTS TESTING FLAG
off=0
mouth_coords=[]

cap=cv2.VideoCapture(0)
#cap = cv2.VideoCapture(2) #LogiTech WebCam
#cap=cv2.VideoCapture("rtsp://admin:admin123@192.168.0.10") #camera stream
#cap=cv2.VideoCapture("http://@127.0.0.0:6969/") #Stream from server

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa=contourDetector(predictor,desiredFaceWidth=256)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)


        landmarks = predictor(gray, face)
        #rect=detector(gray,2)                  #causes a delay of 50 ms minimum when uncommented because dual instance of detector is being used
        #faceAligned=fa.align(frame,gray,rect)

        
        mouth_x1=landmarks.part(48).x
        mouth_y1=landmarks.part(48).y
        mouth_x2=landmarks.part(51).x
        mouth_y2=landmarks.part(51).y
        mouth_x3=landmarks.part(54).x
        mouth_y3=landmarks.part(54).y
        mouth_x4=landmarks.part(57).y
        mouth_y4=landmarks.part(57).y

        for n in range(48,58,3):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            mouth_coords.append((x,y))


            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            #send landmark_points to server A
            #send data to server
    
        mouth_json={
        'mouth_x1':mouth_x1,
        'mouth_y1':mouth_y1,
        'mouth_x2':mouth_x2,
        'mouth_y2':mouth_y2,
        'mouth_x3':mouth_x3,
        'mouth_y3':mouth_y3,
        'mouth_x4':mouth_x4,
        'mouth_y4':mouth_y4
        }
        #print(mouth_coords)
        mouth_json_coords=json.dumps(mouth_json)
        headers={
        'Content-type':'application/json',
        'Accept':'text/plain'
        }        
    #print(mouth_json_coords)
    cv2.imshow("Frame", frame)
    #cv2.imshow("Face Aligned",faceAligned)

    key = cv2.waitKey(1)
    if key == 27:
        break
#(48,51,54,57)
