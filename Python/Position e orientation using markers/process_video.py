## This code processes videos of two cameras and calculates the position of the
# target that has has a blue sphere and a orange one.
# The first camera is set to have a lateral sight of the target and the second 
# camera is set to have a sight of the bottom of the target.
# This two images are used to calculate the position of the target in three 
# dimensions using the angles of the projection in each image to correct the 
# distance calculated by trigonometry  

import numpy as np
import cv2
import imutils
import time
import datetime
import math

## Functions
# Calculate the angle betwen to points in degrees
def get_angle(p1, p2):
	return math.atan2(p1[1] - p2[1], p1[0] - p2[0]) * 180/math.pi

# Calculate the angle betwen to points in radians
def get_angle_radians(p1, p2):
	return math.atan2(p1[1] - p2[1], p1[0] - p2[0])

# Calculate the distance between an object and the camera
def distance_to_camera(knownWidth, focalLength, measuredWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / measuredWidth

# Calculate the distance between to points of the image in pixels
def distance_two_points(p1, p2):
	return math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )

# Calculate the point between two points
def median_distance(p1,p2):
	return (int(round((p1[0] + p2[0])/2)), int(round(((p1[1] + p2[1])/2))))

# Calculate the distance 
def distance_from_center(distance, focalLength, measuredWidth):
	return (distance*measuredWidth)/focalLength

# Calculate the position the object
def calc_position(p1,p2,center,D1, D2, fL1 ,fL2):
	return (distance_from_center(D1, fL1, p1[0]-center[0]),
							   distance_from_center(D2, fL2, p2[1]-center[1]),
							   distance_from_center(D1, fL1, p1[1]-center[2]))
							   
## Variables
	
# Initiate output arrays 
out1 = np.zeros((1,10))
out2 = np.zeros((1,10))
out = np.zeros((1,6))

# Distance between the centers of the spheres in centimeters
knownWidth = 2.6 

# Focal Length is calculated as the distance in pixels from a center the sphere
# to the other one times the distance from the camera in centimeters divided by
# the knownWidth, it varies for each camera.
 
focalLength1 =  1371
focalLength2 = 982 

# Define the ranges of the colours 

blueLower = (105,140,40)
blueUpper = (120,255,255)
orangeLower = (6,140,0)
orangeUpper = (20,255,255)

# Read the video files

video1 = cv2.VideoCapture('t1.avi')
video2 = cv2.VideoCapture('t2.avi')

# Set the resolution to full HD
video1.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
video2.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
video1.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
video2.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

# Set the codec and the parameters to write the processed video file
fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
video1_p = cv2.VideoWriter('track1_p.avi',fourcc1, 24.0, (1280,720))
video2_p = cv2.VideoWriter('track2_p.avi',fourcc2, 24.0, (1280,720))

# Wait time to start processing the video
time.sleep(2.0)

## Process video
while True:    
	read1 = video1.read()
	read2 = video2.read()
	frame1 = read1[1]
	frame2 = read2[1]
		
	if (frame1 is None) or (frame2 is None):
		break

	blurred1 = cv2.medianBlur(frame1,5) 
	blurred2 = cv2.medianBlur(frame2,5) 

	hsv1 = cv2.cvtColor(blurred1, cv2.COLOR_BGR2HSV)
	hsv2 = cv2.cvtColor(blurred2, cv2.COLOR_BGR2HSV)
	
	# Create the masks for each colour in camera 1
	maskB_1 = cv2.inRange(hsv1, blueLower, blueUpper)
	maskB_1 = cv2.erode(maskB_1, None, iterations=2)
	maskB_1 = cv2.dilate(maskB_1, None, iterations=2)

	maskO_1 = cv2.inRange(hsv1, orangeLower, orangeUpper)
	maskO_1 = cv2.erode(maskO_1, None, iterations=2)
	maskO_1 = cv2.dilate(maskO_1, None, iterations=2)
	
	# Create the masks for each colour in camera 2
	maskB_2 = cv2.inRange(hsv2, blueLower, blueUpper)
	maskB_2 = cv2.erode(maskB_2, None, iterations=2)
	maskB_2 = cv2.dilate(maskB_2, None, iterations=2)

	maskO_2 = cv2.inRange(hsv2, orangeLower, orangeUpper)
	maskO_2 = cv2.erode(maskO_2, None, iterations=2)
	maskO_2 = cv2.dilate(maskO_2, None, iterations=2)
	
	# Extract the contours of the masks in camera 1	
	cntsB_1 = cv2.findContours(maskB_1.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cntsB_1 = imutils.grab_contours(cntsB_1)
	centerB_1 = None

	cntsO_1 = cv2.findContours(maskO_1.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cntsO_1 = imutils.grab_contours(cntsO_1)
	centerO_1 = None

	# Extract the contours of the masks in camera 2	
	cntsB_2 = cv2.findContours(maskB_2.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cntsB_2 = imutils.grab_contours(cntsB_2)
	centerB_2 = None

	cntsO_2 = cv2.findContours(maskO_2.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cntsO_2 = imutils.grab_contours(cntsO_2)
	centerO_2 = None

	# Find the biggest contour and get the center of the spheres
	if len(cntsB_1) > 0: 
		cB_1 = max(cntsB_1, key=cv2.contourArea)
		((xB_1, yB_1), radiusB_1) = cv2.minEnclosingCircle(cB_1)
		MB_1 = cv2.moments(cB_1)
		centerB_1 = (int(MB_1["m10"] / MB_1["m00"]),
               int(MB_1["m01"] / MB_1["m00"]))
		
		if radiusB_1 > 10:
			cv2.circle(frame1, (int(xB_1), int(yB_1)), int(radiusB_1),
				(0, 255, 255), 2)
			cv2.circle(frame1, centerB_1, 5, (0, 0, 255), -1)
			
	else:
		xB_1, yB_1 = 0,0
	 
	if len(cntsO_1) > 0:
		cO_1 = max(cntsO_1, key=cv2.contourArea)
		((xO_1, yO_1), radiusO_1) = cv2.minEnclosingCircle(cO_1)
		MO_1 = cv2.moments(cO_1)
		centerO_1 = (int(MO_1["m10"] / MO_1["m00"]),
               int(MO_1["m01"] / MO_1["m00"]))
		
		if radiusO_1 > 10:
			cv2.circle(frame1, (int(xO_1), int(yO_1)), int(radiusO_1),
				(0, 255, 255), 2)
			cv2.circle(frame1, centerO_1, 5, (0, 0, 255), -1)
	else:
		  xO_1, yO_1 = 0,0

	if len(cntsB_2) > 0:  
		cB_2 = max(cntsB_2, key=cv2.contourArea)
		((xB_2, yB_2), radiusB_2) = cv2.minEnclosingCircle(cB_2)
		MB_2 = cv2.moments(cB_2)
		centerB_2 = (int(MB_2["m10"] / MB_2["m00"]),
               int(MB_2["m01"] / MB_2["m00"]))
		
		if radiusB_2 > 10:
			cv2.circle(frame2, (int(xB_2), int(yB_2)), int(radiusB_2),
				(0, 255, 255), 2)
			cv2.circle(frame2, centerB_2, 5, (0, 0, 255), -1)
	else:
		  xB_2, yB_2 = 0,0
		
	if len(cntsO_2) > 0:		
		cO_2 = max(cntsO_2, key=cv2.contourArea)
		((xO_2, yO_2), radiusO_2) = cv2.minEnclosingCircle(cO_2)
		MO_2 = cv2.moments(cO_2)
		centerO_2 = (int(MO_2["m10"] / MO_2["m00"]),
               int(MO_2["m01"] / MO_2["m00"]))
		
		if radiusO_2 > 10:
			cv2.circle(frame2, (int(xO_2), int(yO_2)), int(radiusO_2),
				(0, 255, 255), 2)
			cv2.circle(frame2, centerO_2, 5, (0, 0, 255), -1)
			
	else:
		  xO_2, yO_2 = 0,0

	# Calculate the angles and distances
	if cntsB_1 and cntsO_1:
		alpha1 = get_angle(centerB_1, centerO_1)        
		alpha1R = get_angle_radians (centerB_1, centerO_1)
		distanceBlueOrange1 = distance_two_points(centerB_1, centerO_1)
		center_obj1 = median_distance(centerB_1,centerO_1)
		cv2.circle(frame1, center_obj1, 5, (0, 0, 255), -1)
		
	else:
		alpha1 = 0.0
		centimeters1 = 0.		
		distanceBlueOrange1 = 0. 
		center_obj1 = (0,0)
	
	if cntsB_2 and cntsO_2:     
		alpha2 = get_angle(centerB_2, centerO_2)
		alpha2R = get_angle_radians (centerB_2, centerO_2) 
		distanceBlueOrange2 = distance_two_points(centerB_2, centerO_2) 
		center_obj2 = median_distance(centerB_2, centerO_2)
		cv2.circle(frame2, center_obj2, 5, (0, 0, 255), -1)
		
	else:
		alpha2 = 0.0
		centimeters2 = 0.		
		distanceBlueOrange2 = 0.
		center_obj2 = (0,0)
        
		
#  Calculate the distance of the object from the camera,
# considering the correction needed due to the angle of projection
	if cntsB_1 and cntsO_1 and cntsB_2 and cntsO_2:
		centimeters1 = distance_to_camera(knownWidth*math.cos(alpha2R),
                                    focalLength1,distanceBlueOrange1)  
		centimeters2 = distance_to_camera(knownWidth*math.cos(alpha1R),
                                    focalLength2,distanceBlueOrange2)  
		
	else:
		centimeters1 = 0
		centimeters2 = 0
	
	position = calc_position(center_obj1,center_obj2,(640,360,360),
                          centimeters1,centimeters2,focalLength1,focalLength2)     
	
	# Print the text on the video
	
	cv2.putText(frame1, "alpha1 %.2f" % (alpha1) ,(frame1.shape[1] - 500,
					frame1.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)
	cv2.putText(frame1, "distance %.2fcm" % (centimeters1),
					(frame1.shape[1] - 630, frame1.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX,
					2.0, (0, 255, 0), 3)
	cv2.putText(frame2, "alpha2 %.2f" % (alpha2) ,(frame2.shape[1] - 500,
					frame2.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)
	cv2.putText(frame2, "distance %.2fcm" % (centimeters2),
					(frame2.shape[1] - 630, frame2.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX,
					2.0, (0, 255, 0), 3)
	

	cv2.putText(frame1, "x: %.2f" % (position[0]),
					(frame1.shape[1] - 1270, frame1.shape[0] - 600), cv2.FONT_HERSHEY_SIMPLEX,
					2.0, (0, 255, 0), 3)
	cv2.putText(frame1, "y: %.2f" % (position[2]),
					(frame1.shape[1] - 1270, frame1.shape[0] - 500), cv2.FONT_HERSHEY_SIMPLEX,
					2.0, (0, 255, 0), 3)
	cv2.putText(frame1, "z: %.2f" % (position[1]),
					(frame1.shape[1] - 1270, frame1.shape[0] - 400), cv2.FONT_HERSHEY_SIMPLEX,
					2.0, (0, 255, 0), 3)
	
	# Print the origin of the coordinate system
	cv2.circle(frame1, (640,360), 5, (0, 0, 255), -1)
	cv2.circle(frame2, (640,360), 5, (0, 0, 255), -1)

	# Output file
	x = datetime.datetime.now()
	out1 = np.append(out1,[[float(x.strftime('%H')), float(x.strftime('%M')),
						 float(x.strftime('%S')),xB_1, yB_1, xO_1, yO_1,
						 alpha1, distanceBlueOrange1 ,centimeters1]],axis=0)
	out2 = np.append(out2,[[float(x.strftime('%H')), float(x.strftime('%M')),
						 float(x.strftime('%S')),xB_2, yB_2, xO_2, yO_2,
						 alpha2,distanceBlueOrange2 ,centimeters2]],axis=0)
	out = np.append(out,[[float(x.strftime('%H')), float(x.strftime('%M')),
						 float(x.strftime('%S')), position[0], position[1], 
						 position[2]]], axis=0)
	
	cv2.imshow("Lateral", frame1)
	cv2.imshow("Bottom", frame2)
	video1_p.write(frame1)
	video2_p.write(frame2)
	

	key = cv2.waitKey(1) & 0xFF
		
	if key == ord("q"):
		break


video1.release()
video2.release()
video1_p.release()
video2_p.release()

cv2.destroyAllWindows()
np.save('out1_p1', out1)
np.save('out2_p2', out2)
np.save('out',out)



		