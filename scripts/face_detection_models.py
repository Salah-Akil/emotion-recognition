"""
* I will create a simple script to detect faces in images and use different cascades, since I have detection problems with HAAR
"""

import cv2
from mtcnn import MTCNN
import numpy as np
import os


def detection_haar(img_path,scale_factor=1.10,model_label=False):
    # Haar face cascade
    face_cascade = cv2.CascadeClassifier('opencv/data/haarcascade_frontalface_default.xml')

    """
    I read the image and also convert it to grayscale since most of the
    operations in OpenCV are performed on grayscale images
    """
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    """
    * detectMultiScale() -> 
    will detect objects, since we are calling
    it on face cascades that what it will be detecting, and return a list
    of rectangles where inside it believes it there is a face
    ! scaleFactor ->
    Scale factor has to be tweaked based on the image, since some
    faces might closer to the camera they will appear bigger than the faces
    on the back, so scaleFactor is used to compensate for this
    ! minNeighbors ->
    The detection algorithm uses a moving window to detect objects so we use
    minNeighbors since it defines how many objects are detected near the
    current one before it declares the face found
    ! minSize ->
    It sets the size of each windows
    """
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=scale_factor, # If you set to only 1.05 you will an FP (False Positive) on the real madrid image
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Print the number of faces found
    print(f"Found {len(faces)} faces!")

    # Draw a rectangle around the faces on the original image (not grayscaled)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (110, 110, 255), 2)

    # For the final models comparison it's convenient to have the model named use in the top-left corner
    if(model_label):
        cv2.putText(image,  f"HAAR - ScaleFactor: {scale_factor}", (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 110, 255), 1, cv2.LINE_AA)

    # Display the new image with the bounding boxes on
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)

    # I save the processed image (for thesis doc and comparison)
    filename, file_extension = os.path.splitext(img_path)
    processed_image = filename + f"_haar_scale_factor_{scale_factor}.jpg"
    print("Saving processed image as " + processed_image)
    cv2.imwrite(processed_image, image)



def detection_mtcnn(img_path,landmarks=False,model_label=False):
    image = cv2.imread(img_path)
    detector = MTCNN()

    """
    Save faces detected and the face key features in JSON format:
    {
        'box': [399, 47, 34, 44],
        'confidence': 0.9999960660934448,
        'keypoints': {
            'left_eye': (409, 65),
            'right_eye': (425, 64),
            'nose': (417, 74),
            'mouth_left': (411, 82),
            'mouth_right': (424, 82)
        }
    }
    """
    faces = detector.detect_faces(image) #result
    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        cv2.rectangle(image, (x, y), (x1, y1), (110, 110, 255), 2)

        if (landmarks):
            # Display probability
            cv2.putText(image,  str(round(result['confidence'], 2)), ( result['box'][0],  result['box'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            # Draw Facial Landmarks
            cv2.circle(image, result['keypoints']['left_eye'], 5, (255, 255, 255), -1)
            cv2.circle(image, result['keypoints']['right_eye'], 5, (255, 255, 255), -1)
            cv2.circle(image, result['keypoints']['nose'], 5, (255, 255, 255), -1)
            cv2.circle(image, result['keypoints']['mouth_left'], 5, (255, 255, 255), -1)
            cv2.circle(image, result['keypoints']['mouth_right'], 5, (255, 255, 255), -1)

    # For the final models comparison it's convenient to have the model named use in the top-left corner
    if(model_label):
        cv2.putText(image,  "MTCNN", (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 110, 255), 1, cv2.LINE_AA)


    # Display the new image with the bounding boxes on
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)

    # I save the processed image (for thesis doc and comparison)
    filename, file_extension = os.path.splitext(img_path)
    processed_image = filename + f"_mtcnn_landmarks_{landmarks}.jpg"
    cv2.imwrite(processed_image, image)




def detection_dnn(img_path, min_confidence_score=0.5,model_label=False):
    """
    DNN models
    .caffemodel -> contain the weights for the actual layers
    .protox -> defines the model architecture (basically the layers)
    """
    modelFile = "opencv/data/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "opencv/data/deploy.prototxt"

    # Load the serialized model form the files
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    """
    Read the image and create an input blob for the image by
    changing its size to 300 by 300 pixels and then normalizing it
    * -> In order to try to get the best result possible we have to
    *       run the model on BGR images that have size of (300 by 300 pixels)
    *       and that we apply a mean substraction of (blue=104,green=117,red=123)
    """
    image = cv2.imread(img_path)
    h, w = image.shape[:2] # get height and with from image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
    (300, 300), (104.0, 117.0, 123.0))


    # We now pass the blob through the network adn get back the results
    net.setInput(blob)
    faces = net.forward()

    # For the final models comparison it's convenient to have the model named use in the top-left corner
    if(model_label):
        cv2.putText(image,  f"DNN - Min Confidence Score: {min_confidence_score}", (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 110, 255), 1, cv2.LINE_AA)

    """
    Loop through the results
    1 -> extract the confidence (probability)
    2 -> get rid of the detection that have values lower than the
        min_confidence_score we set
    3 -> compute the (x, y)-coordinates of the bounding box for the face
    4 -> finally draw the bounding box
    """
    for i in range(faces.shape[2]):
        # 1
        confidence = faces[0, 0, i, 2]

        # 2
        if confidence > min_confidence_score:
            # 3
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # 4
            cv2.rectangle(image, (startX, startY), (endX, endY), (110, 110, 255), 2)


    # Display the new image with the bounding boxes on
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)

    # I save the processed image (for thesis doc and comparison)
    filename, file_extension = os.path.splitext(img_path)
    processed_image = filename + f"_dnn_min_confidence_score_{min_confidence_score}.jpg"
    cv2.imwrite(processed_image, image)