# DNN Face Detector in OpenCV

DNN (Deep Neural Network) Face Detector is a Caffe model based on the Single Shot Multibox Detector (SSD).
DNN FC uses the ResNet-10 architecture as its backbone and it is part of the OpenCV deep neural network module (there is also a Tensorflow version).

## Single Shot Multibox Detector

SSD was first prosed by Wei Liu et al. in a paper titled "SSD: Single Shot MultiBox Detector" back in 2016. SSD was described as a method for detecting objects in images using a single deep neural network. GitHub [Repo](https://github.com/weiliu89/caffe/tree/ssd).

The name of the method itself can be divided in:

- **Single Shot** &rarr; meaning that the task of object detection is perform in a single forward pass of the network;
- **MultiBox** &rarr; is the name of the technique used for the task of bounding box regression;
- **Detector** &rarr; meaning the model will detect and classify objects (such as faces).

### Model

The Single Shot Multibox Detector model is based on a feed-forward CNN which creates a fixed-size collection of bounding boxes and score values for representing the presence of object class instances. This is followed by a NMS (Non-Maximum Suppression) in order to merge highly overlapped instances and produce the final detection.

Hence we can divide the model in two sections:

1. Extract feature maps
2. Apply convolutional filter to detect object

The first section, also called ***base network*** contains layers based on the VGG-16 architecture since it's the best architecture model for high quality image classification.

<p align="center">
    <img width="470" height="276" src="https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/vgg_architecture.png?raw=true">
</p>

<p align="center">
    <img src="https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/dnn_vgg_boxed.png?raw=true">
</p>


The task of the VGG model is to extract feature maps. The convolutional layer represented in the image above (Conv4_3) is used for object detection, and is composed of 38x38 cells and it will output 4 object predictions.

<p align="center">
    <img width="450" height="474" src="https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/dnn_vgg_prediction.png?raw=true">
</p>

Each of these prediction is made of a boundary box and scores for each class (we might be training the model on many classes such types of cars) with an additional extra class for "no object". The highest score is set as the bounded object class.

Keep in mind that the VGG-16 architecture will help improve the final results by making use of transfer learning. The only difference SSD takes is that it truncates the VGG-16 classification layers (fully connected layers), and replaces them with a set of auxiliary convolutional layers (from FC6 to the end) in order to extract features at different scales and steadily decrease the size of the input to each subsequent layers.

<p align="center">
    <img src="https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/ssd_architecture.png?raw=true">
</p>


### NMS

The last step of the SSD model is the use of a non-maximum suppression in order to reduce the number of bounding boxes.

<p align="center">
    <img src="https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/dnn_nms.png?raw=true">
</p>


## DNN Code Implementation

Let's now see how we can implement DNN Face Detector in our code.

### Installation & Dependencies

DNN-Caffee need OpenCV in order to run which we can install with `pip`:

```bash
pip install opencv-python
```

Or in case we are running a Conda environment we can use `conda` command:

```bash
conda install -c conda-forge opencv
```

We also need the DNN models (Available in the GitHub repo for this project):

- ***.caffemodel*** &rarr; contain the weights for the actual layers
- ***.protox*** &rarr; defines the model architecture (basically the layers)

### Face Detection Script

The python function we are about to code will take in two parameters as input:

- **img_path** &rarr; the path of the image we want to perform the face detection on
- **min_confidence_score** &rarr; float value between 0 and 1 that base threshold of face detection accuracy.

The function will take the image path, read the image, perform the face detection task, and save as output a copy of the original image with possible bounding boxes that contain the faces detected that have an accuracy greater than the threshold passed in as a parameter.

```python
import cv2
import os

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
```
