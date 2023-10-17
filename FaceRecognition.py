#import pkg_resources
#pkg_resources.require("numpy==`1.21.0")
#import numpy
#import sys
import threading
import cv2
#from deepface import DeepFace
#import random

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# My face
#reference_image = cv2.imread('Front.jpeg')

# My face
#reference_image = cv2.imread('Front.jpeg')

match = False

#matching function -- this needs work
def is_image_match(imgVideo, reference_image, threshold=0.5):
    global match
    
    # Convert the live frame and reference image to grayscale for template matching
    live_frame_gray = cv2.cvtColor(imgVideo, cv2.COLOR_BGR2GRAY)
    reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(live_frame_gray, reference_image_gray, cv2.TM_CCOEFF_NORMED)

    # Find the best match location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Check if the maximum correlation value is above the threshold
    if max_val >= threshold:
        match = True
    else:
        match = False

count = 0

# Video Capture
while 1:
    reference_image = cv2.imread('Front.jpeg')
    ret, imgVideo = capture1.read()  
    
    if ret:
        if count % 24 == 0:
            
            threading.Thread(target = is_image_match(imgVideo, reference_image), args = (imgVideo.copy(),)).start()

            # convert to gray scale of each frames 
            gray = cv2.cvtColor(imgVideo, cv2.COLOR_BGR2GRAY) 

            # Detects faces of different sizes in the input image 
            faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
            
        count += 1
        
        if match == True:
            for (x,y,w,h) in faces: 
                # To draw a rectangle in a face  
                cv2.rectangle(imgVideo,(x,y),(x+w,y+h),(255,255,0),2)  
                roi_gray = gray[y:y+h, x:x+w] 
                roi_color = imgVideo[y:y+h, x:x+w] 
                cv2.putText(imgVideo, "Face is ME", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            # Display an image in a window 
            cv2.imshow('img',imgVideo) 
        else:
            for (x,y,w,h) in faces: 
                # To draw a rectangle in a face  
                cv2.rectangle(imgVideo,(x,y),(x+w,y+h),(255,255,0),2)  
                roi_gray = gray[y:y+h, x:x+w] 
                roi_color = imgVideo[y:y+h, x:x+w] 
                cv2.putText(imgVideo, "Face is NOT ME", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            # Display an image in a window 
            cv2.imshow('img',imgVideo) 
            
    # Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break

# Close the window 
capture1.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()
