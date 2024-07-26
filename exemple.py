"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2 as cv
from gazeTracker import GazeTracking

gaze = GazeTracking()
webcam = cv.VideoCapture(0)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv.putText(frame, text, (90, 60), cv.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv.imshow("Demo", frame)

    if cv.waitKey(1) == 27:
        break
   
webcam.release()
cv.destroyAllWindows()
