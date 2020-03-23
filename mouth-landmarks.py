# download "shape_predictor_68_face_landmarks.dat" from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

import cv2
import itertools
import scipy.interpolate
from skimage import color
#import zmq
import numpy as np
import dlib
#import websocket
#from websocket import create_connection
#import json

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

lip_x=[]
lip_y=[]
red_l=0.0
blue_l=0.0
green_l=0.0


def draw_curve(points):
    #draws a curve around interpolated path
    #points is an array with all lip-coords inside them (each lip coord is an array of x and y coordinate)
    debug=0
    x_pts=[]
    y_pts=[]
    curvex=[]
    curvey=[]
    debug+=1
    for point in points:
        x_pts.append(point[0])
        y_pts.append(point[1])
    curve=scipy.interpolate.interp1d(x_pts,y_pts,'cubic')
    
    if debug==1 or debug==2:
        for i in np.arange(x_pts[0],x_pts[len(x_pts)-1]+1,1):
            curvex.append(i)
            curvey.append(int(curve(i)))
    else:
        for i in np.arange(x_pts[len(x_pts)-1]+1,x_pts[0],1):
            curvex.append(i)
            curvey.append(int(curve(i)))
    
    return curvex,curvey

def fill_lip_lines(outer,inner):
    #fills outlines of lips with a colour
    outer_curve=zip(outer[0],outer[1])
    inner_curve=zip(inner[0],inner[1])
    count=len(inner[0])-1
    print(inner[0])
    print(inner[0][-1])
    print(' ')
    #print(inner[1])
    print(count)
    last_inner = [inner[0][-1], inner[1][-1]]
    #last_inner = [inner[0][count], inner[1][count]]
    for o_point,i_point in itertools.zip_longest(outer_curve,inner_curve,fillvalue=last_inner):
        line=scipy.interpolate.interp1d([o_point[0],i_point[0]],[o_point[1],i_point[1]],'linear')
        xpoints=list(np.arange(o_point[0],i_point[0],1))
        lip_x.extend(xpoints)
        lip_y.extend([int(point) for point in line(xpoints)])
    #return

def fill_lip_solid(outer,inner):
    #fills solid colour inside the outlines
    inner[0].reverse()
    inner[1].reverse()
    outer_curve=zip(outer[0],outer[1])
    inner_curve=zip(inner[0],inner[1])
    points=[]
    for point in outer_curve:
        points.append(np.array(point,dtype=np.int32))
    for point in inner_curve:
        points.append(np.array(point,dtype=np.int32))
    points=np.array(points,dtype=np.int32)
    red_l=int(red_l)
    green_l=int(green_1)
    blue_l=int(blue_l)
    cv2.fillPoly(frame,[points],(red_l,green_l,blue_l))

def smoothen_colour(img,outer,inner):
    img.height,img.width=img.shape[:2]
    img1=img.copy()
    outer_curve=zip(outer[0],outer[1])
    inner_curve=zip(inner[0],inner[1])
    x_points=[]
    y_points=[]
    for point in outer_curve:
        x_points.append(point[0])
        y_points.append(point[1])
    for point in inner_curve:
        x_points.append(point[0])
        y_points.append(point[1])
    img_base=np.zeros((img.height,img.width))
    cv2.fillConvexPoly(img_base,(81,81),0)
    img_blur_3d=np.ndarray([img.height,img.width,3],dtype='float')
    img_blur_3d[:,:,0]=img_mask
    img_blur_3d[:,:,1]=img_mask
    img_blur_3d[:,:,2]=img_mask
    img1=(img_blur_3d*img*0.7+(1-img_blur_3d*0.7)*img1).astype('uint8')

def add_colour(img,intensity):
    #adds base color to all points on lips, at mentioned intensity
    val = color.rgb2lab((img[lip_y,lip_x] / 255.).reshape(len(lip_y),1,3)).reshape(len(lip_y),3)
    l_val,a_val,b_val=np.mean(val[:,0]),np.mean(val[:,1]),np.mean(val[:,2])
    l1_val, a1_val, b1_val = color.rgb2lab(np.array((red_l / 255.,green_l / 255.,blue_l / 255.)).reshape(1, 1, 3)).reshape(3,)
    l_final, a_final, b_final = (l1_val - l_val) * \
    intensity, (a1_val - a_val) * \
    intensity, (b1_val - b_val) * intensity

    val[:,0]=np.clip(val[:,0]+l_final,0,100)
    val[:,1]=np.clip(val[:,1]+a_final,-127,128)
    val[:,2]=np.clip(val[:,2]+b_final,-127,128)
    img[lip_y,lip_x]=color.lab2rgb(val.reshape(len(lip_y),1,3)).reshape(len(lip_y),3)*255

def get_points_lips(lip_points):
    #gets points of lips (lip_points is an array of lip coords)
    uol=[]
    uil=[]
    lol=[]
    lil=[]
    for i in range(0,14,2):
        uol.append([int(lips_points[i]),int(lips_points[i+1])])
    for i in range(12,24,2):
        lol.append([int(lips_points[i]),int(lips_points[i+1])])
    lol.append([int(lips_points[0]),int(lips_points[1])])
    for i in range(24,34,2):
        uil.append([int(lips_points[i]),int(lips_points[i+1])])
    for i in range(32,40,2):
        lil.append([int(lips_points[i]),int(lips_points[i+1])])
    lil.append([int(lips_points[24]),int(lips_points[25])])

    return uol,uil,lol,lil

def get_curves_lips(uol,uil,lol,lil):
    #get outline of lips
    uol_curve=draw_curve(uol)
    uil_curve=draw_curve(uil)
    lol_curve=draw_curve(lol)
    lil_curve=draw_curve(lil)

    return uol_curve,uil_curve,lol_curve,lil_curve

def fill_colour(img,uol_c,uil_c,lol_c,lil_c):
    #fill color in lips
    fill_lip_lines(uol_c,uil_c)
    fill_lip_lines(lol_c,lil_c)
    add_colour(img,1)
    fill_lip_solid(uol_c,uil_c)
    fill_lip_solid(lol_c,lil_c)
    smoothen_colour(img,uol_c,uil_c)
    smoothen_colour(img,lol_c,lil_c)


#websocket.enableTrace(True)
#ws=create_connection('ws://') #address goes here

cap=cv2.VideoCapture(0)

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

            #cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            #send landmark_points to server A

            #mouth landmarks
            mouth_coords=[]
            for j in range(48,68):
                mouth_x=landmarks.part(j).x
                mouth_y=landmarks.part(j).y
                mouth_coords.append((mouth_x,mouth_y))  #These coords will be sent to the next server, usage(?)
                
                cv2.circle(frame,(mouth_x,mouth_y),2,(0,0,255),-1)
                #ws.send(json.dumps([json.dumps({'msg': 'connect', 'version': '1', 'support': mouth_coords})])) 

        #These coords would be used to fill colour into the lips
        face_coords=np.matrix([[p.x,p.y] for p in landmarks.parts()])
        lips=""
        for point in face_coords[48:68]:
            lips+=str(point).replace('[','').replace(']','')+'\n'
        lips=list([point.split() for point in lips.split('\n')])
        #print(lips)
        lips_points=[item for sublist in lips for item in sublist]
        uol,uil,lol,lil=get_points_lips(lips_points)
        uol_c,uil_c,lol_c,lil_c=get_curves_lips(uol,uil,lol,lil)
        lipstick=fill_colour(frame,uol_c,uil_c,lol_c,lil_c)


            
    cv2.imshow("Frame", frame)
    #cv2.imshow("Face Aligned",faceAligned)

    key = cv2.waitKey(1)
    if key == 27:
        break
