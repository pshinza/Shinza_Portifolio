# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 23:28:37 2019

@author: pedro
"""
## Concatenate two full HD videos (vertical)

import numpy as np
import cv2
import imutils

# Read the files
video1 = cv2.VideoCapture('track1_p.avi')
video2 = cv2.VideoCapture('track2_p.avi')

# Select the codec and set parameters for saving the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_two = cv2.VideoWriter('video_processed.mp4',fourcc, 24.0, (640,720))

# Process the video
while True:
    
    read1 = video1.read()
    read2 = video2.read()
    
    frame1 = read1[1]
    frame2 = read2[1]
    if (frame1 is None) or (frame2 is None):
        break    
    
    frame1 = imutils.resize(frame1, width=640)
    frame2 = imutils.resize(frame2, width=640)     
    
    # stack the frames vertically
    frame = np.vstack((frame1,frame2))
        
    video_two.write(frame)

    key = cv2.waitKey(1) & 0xFF
		
    if key == ord("q"):
        break

video_two.release()
video1.release()
video2.release()
cv2.destroyAllWindows()    
    
    
    
        
    
    
    