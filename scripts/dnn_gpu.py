from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
import os
from face_detection_server import *

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
                help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
                help="path to (optional) output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-f", "--face-detection", type=str, default="DNN",
                help="face detection model to use (DNN or MTCNN")
ap.add_argument("-s", "--dataset", type=str, default="kaggle",
                help="dataset model to use (kaggle or ck)")
ap.add_argument("-m", "--model", type=str, default="kaggle_model_v12.h5",
                help="path to the model to use (make sure model coincide with dataset used")
ap.add_argument("-u", "--use-gpu", type=bool, default=False,
                help="boolean indicating if CUDA GPU should be used")
ap.add_argument("-d", "--display", type=int, default=1,
                help="whether or not output frame should be displayed")
args = vars(ap.parse_args())


# declare emotion classes based on the dataset chosen
EMOTIONS_LIST_CK = ["Angry", "Contempt", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
EMOTIONS_LIST_KAGGLE = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
if args["dataset"] == "ck":
    CLASSES = EMOTIONS_LIST_CK
else:
    CLASSES = EMOTIONS_LIST_KAGGLE
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# initialize the video stream and pointer to output video file, then
# start the FPS timer
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
fps = FPS().start()
# loop over the frames from the video stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break
    # resize the frame if you want. for now I will not change it
    # # frame = imutils.resize(frame, width=400)
    # check to see if the output frame should be displayed to our screen
    if args["display"] > 0:
        # show the output frame (you can add a switch to choose which face detection model to use)
        cv2.imshow("Frame", detection_dnn(frame))
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # if an output video file path has been supplied and the video
    # writer has not been initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (frame.shape[1], frame.shape[0]), True)
    # if the video writer is not None, write the frame to the output
    # video file
    if writer is not None:
        writer.write(frame)
    # update the FPS counter
    fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))