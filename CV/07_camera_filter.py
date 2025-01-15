import cv2
import sys
import numpy as np

PREVIEW=0 #Preview Mode
BLUR=1 #Blurring Filter
FEATURES=2 #Corner Feature Detector
CANNY=3 #Canny Edge Detector


feature_params=dict(maxCorners=500,qualityLevel=0.2,minDistance=15,blockSize=9)  #the qualityLevel decide the threshold for which based on response value of corner to be considered as a valid feature
# R<qualityLevel*Rmax

s=0
if len(sys.argv)>1:
    s=sys.argv[1]


image_filter=PREVIEW
alive=True

win_name='Camera Filter'
cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
result=None


source=cv2.VideoCapture(s)

# we will do some processing and send the processed results to the output window
while alive:
    has_frame,frame=source.read()
    if not has_frame:
        break

    frame=cv2.flip(frame,1)

    if image_filter==PREVIEW:
        result=frame
    elif image_filter==CANNY:
        result=cv2.Canny(frame,80,150)  #pixels whose intesity gradients are above the threshold those pixels are sure edge and similarily those below the threshold will be compeletely discarded
        # and those pixels having intensity gradient between the threshold will be considered as candidate edge

    elif image_filter==BLUR:
        result = cv2.blur(frame, (13, 13))  #uses a box filter to blur the image and 13*13 box will be convolved with the image to result in a blurred image 
    
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)  #returns a list of corners based on the paramaters defined in the dictionary
        if corners is not None:
            for x, y in np.float32(corners).reshape(-1, 2):
                cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)  #small green circles to annontate the corners at the location 
    
    cv2.imshow(win_name, result)  #send that to output stream

    key = cv2.waitKey(1)
    if key == ord("Q") or key == ord("q") or key == 27:
        alive = False
    elif key == ord("C") or key == ord("c"):
        image_filter = CANNY
    elif key == ord("B") or key == ord("b"):
        image_filter = BLUR
    elif key == ord("F") or key == ord("f"):
        image_filter = FEATURES
    elif key == ord("P") or key == ord("p"):
        image_filter = PREVIEW

        