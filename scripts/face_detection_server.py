import cv2
from mtcnn import MTCNN
import numpy as np
import os
import tensorflow as tf
import model as mdl


model = mdl.Model("kaggle_model_v12.h5")

def detection_haar(frame,scale_factor=1.10):
    # Haar face cascade
    face_cascade = cv2.CascadeClassifier('../opencv/data/haarcascade_frontalface_default.xml')

    tf_gray_image = tf.image.rgb_to_grayscale(frame)
    cv_gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        cv_gray_image,
        scaleFactor=1.10, # If you set to only 1.05 you will an FP (False Positive) on the real madrid image
        minNeighbors=5,
        minSize=(30, 30)
    )
    # Draw a rectangle around the faces on the original image (not grayscaled)
    for (x, y, w, h) in faces:
        face_detected = tf_gray_image[y:y+h, x:x+w]
        resized_face = tf.image.resize(face_detected, [48,48])
        tf_image = tf.expand_dims(resized_face, axis=0)
        pred = model.predict_emotion_kaggle(tf_image)
        cv2.putText(frame, pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 110, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (110, 110, 255), 2)

    return frame



def detection_mtcnn(frame,landmarks=False):
    detector = MTCNN()

    tf_gray_image = tf.image.rgb_to_grayscale(frame)

    faces = detector.detect_faces(frame) #result
    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h

        face_detected = tf_gray_image[y:y1, x:x1]
        resized_face = tf.image.resize(face_detected, [48,48])
        tf_image = tf.expand_dims(resized_face, axis=0)
        pred = model.predict_emotion_kaggle(tf_image)
        cv2.putText(frame, pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 110, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x1, y1), (110, 110, 255), 2)

        if (landmarks):
            # Display probability
            cv2.putText(frame,  str(round(result['confidence'], 2)), ( result['box'][0],  result['box'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            # Draw Facial Landmarks
            cv2.circle(frame, result['keypoints']['left_eye'], 5, (255, 255, 255), -1)
            cv2.circle(frame, result['keypoints']['right_eye'], 5, (255, 255, 255), -1)
            cv2.circle(frame, result['keypoints']['nose'], 5, (255, 255, 255), -1)
            cv2.circle(frame, result['keypoints']['mouth_left'], 5, (255, 255, 255), -1)
            cv2.circle(frame, result['keypoints']['mouth_right'], 5, (255, 255, 255), -1)

    return frame




def detection_dnn(frame, min_confidence_score=0.5):
    modelFile = "../opencv/data/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "../opencv/data/deploy.prototxt"

    # Load the serialized model form the files
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    tf_gray_image = tf.image.rgb_to_grayscale(frame)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))


    # We now pass the blob through the network adn get back the results
    net.setInput(blob)
    faces = net.forward()

    for i in np.arange(0, faces.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = faces[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        # if confidence > args["confidence"]:
        if confidence > 0.6:
            # extract the index of the class label from the
            # `faces`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(faces[0, 0, i, 1])
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # draw the prediction on the frame
            face_detected = tf_gray_image[startY:endY, startX:endX]
            resized_face = tf.image.resize(face_detected, [48,48])
            tf_image = tf.expand_dims(resized_face, axis=0)
            pred = model.predict_emotion_kaggle(tf_image)
            cv2.putText(frame, pred, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 110, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (110, 110, 255), 2)


    return frame