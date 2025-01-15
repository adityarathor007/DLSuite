# streaming output from the webcam to the output display 
import cv2
import sys

s = 0
if len(sys.argv) > 1:  #to override the default value
    s = sys.argv[1]

source = cv2.VideoCapture(s)  #cv2.VideoCapture is designed to handle direct access to video files or streams (such as local files, cameras, or video URLs pointing to raw video streams)

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27: # Escape Key pressed or not
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)