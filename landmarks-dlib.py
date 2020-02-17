import cv2
import numpy as np
import dlib
import json

class FaceAligner:
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

    #TODO:
    #wrapAffine to stop the neck points from moving continuously
    #RotationalMatrix to gurantee correct neck points when head turns



cap=cv2.VideoCapture(0)
#cap = cv2.VideoCapture(2) #LogiTech WebCam

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
            #send landmark_points to server A

            eye_x1=landmarks.part(39).x
            eye_y1=landmarks.part(39).y
            eye_x2=landmarks.part(42).x
            eye_y2=landmarks.part(42).y
            nose_x=landmarks.part(27).x
            nose_y=landmarks.part(27).y
            chin_x1=landmarks.part(5).x
            chin_y1=landmarks.part(5).y
            chin_x2=landmarks.part(11).x
            chin_y2=landmarks.part(11).y
            #cv2.circle(frame,(nose_x,nose_y),4,(0,255,0),-1)
            eyeDistance=abs(eye_x1-eye_x2)
            eyeNoseYDistance=((eye_y1+eye_y2)/2-nose_y)

            neck_x1=chin_x1
            neck_x2=chin_x2
            neck_y1=neck_y2=chin_y2+30
            
            #cv2.circle(frame,(chin_x2,chin_y2),4,(0,255,0),-1)
            cv2.circle(frame,(neck_x2,neck_y2),4,(0,255,0),-1)
            cv2.circle(frame,(neck_x1,neck_y1),4,(0,255,0),-1)

            #send data to server
            json_neck_coords={
            "neck_x1":neck_x1,
            "neck_y1":neck_y1,
            "neck_x2":neck_x2,
            "neck_y2":neck_y2
            }

            json_dump=json.dump(json_neck_coords)
            
            
        '''
        #left eye coordinates
        leftEyePts=[]
        for i in range(36,42):
            cerx=landmarks.part(i).x
            cery=landmarks.part(i).y
            leftEyePts.append((cerx,cery))
        #right eye coordinates
        rightEyePts=[]
        for i in range(42,48):
            rerx=landmarks.part(i).x
            rery=landmarks.part(i).y
            #cv2.circle(frame,(rerx,rery),4,(0,0,255),-1)
            rightEyePts.append((rerx,rery))
        #angle calc
        leftEyeCenter=leftEyePts.mean(axis=0).astype('int')
        rightEyeCenter=rightEyePts.mean(axis=0).astype('int')

        dy=rightEyeCenter[1]-leftEyeCenter[1]
        dx=rightEyeCenter[0]-leftEyeCenter[0]
        angle=np.degrees(np.arctan2(dy,dx))-180
        cv2.putText(frame,angle,(0,0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        '''

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

