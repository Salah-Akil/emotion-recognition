# MTCNN

MTCNN stands for Multi-Task Cascaded Convolutional Networks, and is a framework proposed by Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li in their 2016 paper named *"Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks"*, with the objective to perform face detection and face alignment.

The process MTCNN takes is completely different form the Haar method proposed by Viola-Jones. This is because MTCNN uses deep learning in order to achieve impressive performance on face detection and face alignment. In particular, the framework uses a cascaded-structure with three stages of carefully designed deep convolutional networks that predict face and landmark such as eyes, nose and mouth. These stages are divided in:

- Stage 1 -> which makes use of a shallow CNN to quickly produce candidate windows;
- Stage 2 -> use of a more complex CNN in order to refine the windows to reject a large number of non-faces windows;
- Stage 3 -> use of a more powerful CNN to refine the result and output facial landmarks.

## 1. MTCNN Stages

All three stages of the MTCNN take as input a pyramid of scaled images of the original image to process.

![alt text](images\mtcnn_pyramid.png "Pyramid Image Representation")

### 1.1 Stage 1 (P-Net)

The first stage exploits a fully convolutional network called Proposal Network (**P-Net**). The main difference between a convolutional neural network and a fully convolutional one is that the FCN does not make use of a dense layers as part of the architecture.

The P-Net takes as input.

The P-Net is used to calculate the candidate windows and their bounding box regression vectors. Bounding box regression is a mainstream technique to refine or predict localization boxes. Usually bounding box regressors are trained to regress from either region proposals or fixed anchor boxes to nearby bounding boxes of a pre-defined target object classes. Once this estimated bounding boxes are calculated they are then used to calibrate the candidate windows. After doing this, a non-maximum suppression (NSM) is used to merge highly overlapped candidates.

![alt text](images\p_net.jpeg "P-Net")

The first stage will give then as output all the candidate windows after they are refined in order to downsize the number of candidates.

![alt text](images\mtcnn_stage_1.png "Stage 1")

### 1.2 Stage 2 (R-Net)

The second stage takes all the candidates and feeds them as input to another CNN, called Refine Network (R-Net), which is used to eliminate even more false candidates. It then performs calibration with bounding box regression and makes use again of NMS to merge overlapped candidates.
The R-Net used is a true CNN and not a FCN like the one used in the first stage, since it makes use of a dense layers at the last stage of the network architecture.

![alt text](images\r_net.jpeg "R-Net")

As we can see in the image above, R-Net will output:

- a face classification (wether the face is a face or not);
- a 4 element vector (which makes up the bounding box for the face);
- and a 10 element vector (which contains facial landmarks localization).

![alt text](images\mtcnn_stage_2.png "Stage 2")

### 1.3 Stage 3 (O-Net)

The Output Network is the last stage and is very similar to the second stage, in fact it uses again NMS to further reduce the candidates to only one and 5 facial landmarks for:

- Left Eye
- Right Eye
- Nose
- Left Mouth Corner
- Right Mouth Corner

![alt text](images\mtcnn_stage_3.png "Stage 3")

The O-Net is composed of 3 convolutional layers having 3x3 filters and 3x3 max-pooling. At the end of the network we have a simple 2x2 filter convolutional layer and a fully connected layer.

![alt text](images\o_net.jpeg "O-Net")

## 2. MTCNN Code Implementation

Let's now see how we can implement MTCNN face detection in our code.

### 2.1 Installation & Dependencies

MTCNN can be installed either via `pip` (we also need to install OpenCV):

```bash
pip install mtcnn
pip install opencv-python
```

Or in case we are running a Conda environment we can use `conda` command:

```bash
conda install -c conda-forge mtcnn
conda install -c conda-forge opencv
```

### 2.2 Face Detection Script

The python function we are about to code will take in two parameters as input:

- **img_path** &rarr; the path of the image we want to perform the face detection on
- **landmarks** &rarr; boolean value representing if we want to display the facial landmarks or not (set to False by default)

The function will take the image path, read the image, perform the face detection task, and save as output a copy of the original image with possible bounding boxes that contain the faces detected.

```python
import cv2
from mtcnn import MTCNN
import os

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
```
