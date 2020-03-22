from scipy.interpolate import interp1d
from pylab import *
from skimage import io, color
import cv2
import numpy as np
import dlib

r,g,b = (00.,00.,200.)		#lipstick color

up_left_end = 3
up_right_end = 5
mouth=[]
#mouth=np.array(mouth)

def inter(lx=[],ly=[],k1='quadratic'):
	unew = np.arange(lx[0], lx[-1]+1, 1)
	f2 = interp1d(lx, ly, kind=k1)
	return (f2,unew)

cap=cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame=cv2.resize(frame,(700,700))

	faces = detector(gray)
	for face in faces:
		x1 = face.left()
		y1 = face.top()
		x2 = face.right()
		y2 = face.bottom()
		#cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
		
		landmarks = predictor(gray, face)

		coord1=landmarks.part(48).x
		coord2=landmarks.part(48).y
		coord3=landmarks.part(49).x
		coord4=landmarks.part(49).y
		mouth=np.array([[coord1,coord2],[coord3,coord4]])
		for n in range(50,62):
			x = landmarks.part(n).x
			y = landmarks.part(n).y
			mouth=np.append(mouth,[[x,y]],axis=0)	
			
		
			#mouth=np.loadtxt("/home/rajat/Desktop/CamCann/d.txt")
			points =  np.floor(mouth)
			print(points)
			point_out_x = np.array((points[:len(points)//2][:,0]))
			point_out_y = np.array(points[:len(points)//2][:,1])
			point_in_x = (points[len(points)//2:][:,0])
			point_in_y = points[len(points)//2:][:,1]

					
			#figure()
			#im = imread('Input.jpg')

			# Code for the curves bounding the lips
		o_u_l = inter(point_out_x[:up_left_end],point_out_y[:up_left_end])
		o_u_r = inter(point_out_x[up_left_end-1:up_right_end],point_out_y[up_left_end-1:up_right_end])
		o_l = inter([point_out_x[0]]+point_out_x[up_right_end-1:][::-1].tolist(),[point_out_y[0]]+point_out_y[up_right_end-1:][::-1].tolist(),'cubic')
		i_u_l = inter(point_in_x[:up_left_end],point_in_y[:up_left_end])
		for i in range()
		print(point_out_x)
		print(point_out_y)
		print(o_u_l)	
		#print(inter(point_in_x[0:3],point_in_y[0:3]))	
		i_u_r = inter(point_in_x[up_left_end-1:up_right_end],point_in_y[up_left_end-1:up_right_end])
			
		i_l = inter([point_in_x[0]]+point_in_x[up_right_end-1:][::-1].tolist(),[point_in_y[0]]+point_in_y[up_right_end-1:][::-1].tolist(),'cubic')

		x = []	#will contain the x coordinates of points on lips
		y = []  #will contain the y coordinates of points on lips

		def ext(a,b,i):
			a=int(np.round(a))
			b=int(np.round(b))
			#print(arange(a,b,1))
			x.extend(arange(a,b,1).tolist())
			y.extend((ones(b-a)*i).tolist())

		for i in range(int(o_u_l[1][0]),int(i_u_l[1][0]+1)):
			ext(o_u_l[0](i), o_l[0](i)+1, i)

		for i in range(int(i_u_l[1][0]),int(o_u_l[1][-1]+1)):
			ext(o_u_l[0](i),i_u_l[0](i)+1,i)
			ext(i_l[0](i),o_l[0](i)+1,i)

		for i in range(int(i_u_r[1][-1]),int(o_u_r[1][-1]+1)):
			ext(o_u_r[0](i),o_l[0](i)+1,i)

		for i in range(int(i_u_r[1][0]),int(i_u_r[1][-1]+1)):
			ext(o_u_r[0](i),i_u_r[0](i)+1,i)
			ext(i_l[0](i),o_l[0](i)+1,i)

		# Now x and y contains coordinates of all the points on lips

		int_x=[int(z) for z in x]
		int_y=[int(z) for z in y]

		#int_x = map(int,x)
		#int_y = map(int,y)

		#print(x)
		#print("\n")
		#print(int_y)
		#print(im[int_x,:])
		#val = color.rgb2lab((frame[x,y]/255.).reshape(len(x),1,3)).reshape(len(x),3)
		#print(frame.shape)
		frame=cv2.resize(frame,(700,700))
		val = color.rgb2lab((frame[int_x,int_y]/255.).reshape(len(int_x),1,3)).reshape(len(int_x),3)
		L,A,B = mean(val[:,0]),mean(val[:,1]),mean(val[:,2])
		L1,A1,B1 = color.rgb2lab(np.array((r/255.,g/255.,b/255.)).reshape(1,1,3)).reshape(3,)
		ll,aa,bb = L1-L,A1-A,B1-B
		val[:,0] += ll
		val[:,1] += aa
		val[:,2] += bb
		
		#val[:,0] = np.clip(val[:,0], 0, 100)
		#val[:,1] = np.clip(val[:,1], -127, 128)
		#val[:,2] = np.clip(val[:,2], -127, 128)

		frame[int_x,int_y] = color.lab2rgb(val.reshape(len(int_x),1,3)).reshape(len(int_x),3)*255
		gca().set_aspect('equal', adjustable='box')
	
	cv2.imshow("frame",frame)

	key = cv2.waitKey(1)
	if key == 27:
		break
