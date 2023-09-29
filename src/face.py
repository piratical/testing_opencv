#!/home/edtrager/C/ComputerVision/cenv/bin/python

import sys
import os
import cv2

# Get command line argument:
arg_count = len(sys.argv)
if(arg_count)!=2:
    print('Please pass a file name to pass on the command line.')
    sys.exit(0)

# get here if a file name was passed:
image_file= sys.argv[1]


# Load pre-trained frontal face data (haar cascade algorithm):
trained_face_data = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

# Choose an image to detect faces in:
img    = cv2.imread(image_file)
gs_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Detect faces:
face_coordinates = trained_face_data.detectMultiScale(gs_img);



for rect in face_coordinates:
    (x,y,w,h) = rect
    #cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    center = ( x+w//2 , y+h//2 )
    axes = (w//2,h//2)
    #radius = (w+h)//4
    color  = (0,255,0)
    cv2.ellipse(img,center,axes,0,0,360,color,2)

face_count = len(face_coordinates)
plural = face_count!=1
ess = 's' if plural else ''
window_name = f'Detected {face_count} face{ess}'

# Define callback function to close window with mouse click:
def close_all(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONUP:
        # closing all open windows
        cv2.destroyAllWindows()
        print(f'Goodbye!')
        sys.exit(0)
#

cv2.imshow(window_name,img)
cv2.setMouseCallback(window_name,close_all);

# This allows exiting with a key press:
cv2.waitKey(0)


# closing all open windows
cv2.destroyAllWindows()
print(f'Goodbye!')

