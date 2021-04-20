# Face Detection Models Comparison

We have seen how HAAR, MTCNN and DNN work for Face Detection. We also programmed functions to take in as input an image and detect if it contains a face and draw a bounding box around it. Let's now see how they compare to each other in terms of performance and accuracy.

## 1. Dataset

We will compare each model on different types of faces in different situations, the main differences we are interested on are:

- Ethnicity
- Age Group
- Front vs Side Face
- Ornaments Occlusion

We will also use images that contain a high number of faces in order to see how many can be detected.
The total dataset is composed of 70 high resolution photos that have been resized to to about 854x480.

### 1.1 Hypothesis

I think that maybe some models will have problems detecting people with very dark skin or faces in dark environment lighting. Also I believe detecting side faces will be very difficult.

## 2. Comparison

We will code a simple script to take one image as input and maybe additional values such as accuracy threshold, and pass them to the "face detection" functions that we already created for each model.

### 2.1 Code

```python
import face_detection_models as fm
import os

photos = os.listdir("face_comparison_dataset")
for i in photos:
    print(i)
    fm.detection_haar(f"face_comparison_dataset/{i}",scale_factor=1.05,model_label=True)
    fm.detection_haar(f"face_comparison_dataset/{i}",scale_factor=1.10,model_label=True)
    fm.detection_haar(f"face_comparison_dataset/{i}",scale_factor=1.20,model_label=True)
    fm.detection_mtcnn(f"face_comparison_dataset/{i}",model_label=True)
    fm.detection_dnn(f"face_comparison_dataset/{i}",min_confidence_score=0.2,model_label=True)
    fm.detection_dnn(f"face_comparison_dataset/{i}",min_confidence_score=0.4,model_label=True)
    fm.detection_dnn(f"face_comparison_dataset/{i}",min_confidence_score=0.6,model_label=True)
```

## 3. Results

We are now going to see how the different face detector compared on the dataset images, but on this section we will be displaying the result of only 20 selected images (that will be displayed in the following order in the comparisons below):

- photo_01
- photo_02
- photo_04
- photo_05
- photo_06
- photo_08
- photo_09
- photo_13
- photo_14
- photo_21
- photo_22
- photo_25
- photo_34
- photo_35
- photo_40
- photo_41
- photo_48
- photo_49
- photo_55
- photo_61

The results of all the 70 images in the dataset can be found in the GitHub repository for this project in this [folder](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/).


### 3.1 HAAR

Let's quickly see how the haar detector performed and how different values for the `scale_factor` impact detection precision.

1.05 | 1.10 | 1.20
:-------------------------:|:-------------------------:|:-------------------:
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_01_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_01_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_01_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_02_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_02_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_02_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_04_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_04_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_04_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_05_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_05_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_05_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_06_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_06_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_06_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_08_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_08_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_08_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_09_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_09_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_09_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_13_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_13_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_13_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_14_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_14_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_14_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_21_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_21_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_21_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_22_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_22_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_22_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_25_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_25_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_25_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_34_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_34_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_34_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_35_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_35_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_35_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_40_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_40_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_40_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_41_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_41_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_41_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_48_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_48_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_48_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_49_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_49_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_49_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_55_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_55_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_55_haar_scale_factor_1.2.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_61_haar_scale_factor_1.05.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_61_haar_scale_factor_1.1.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_61_haar_scale_factor_1.2.jpg?raw=true)

As we can see in the images above, the HAAR model performed very poorly on 3 categories:

- Sideway faces
- Faces that contain occlusive wear such as glasses or masks
- Faces that are very dark in skin

We can also deduce that the best `scale_factor` to use is `1.05` since it offers the best *high precision/low false positive* average.

### 3.2 MTCNN

For MTCNN we didn't set any threshold to the minimum confidence before declaring a bounding box, everything was left to the default values.

Default Values |
:-------------------------:|
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_01_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_02_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_04_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_05_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_06_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_08_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_09_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_13_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_14_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_21_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_22_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_25_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_34_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_35_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_40_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_41_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_48_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_49_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_55_mtcnn_landmarks_False.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_61_mtcnn_landmarks_False.jpg?raw=true)

The MTCNN face detection model performed very poorly on:

- Sideway faces
- Faces that contain occlusive wear such as glasses or masks

But it performed very good on images containing faces that are far away from the camera.
So for now between HAAR and MTCNN, the latter is clearly more precise than the former.

### 3.3 DNN

The DNN detector was set with different `min_confidence_score` values, let's see how it performed:


0.2 | 0.4 | 0.6
:-------------------------:|:-------------------------:|:-------------------:
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_01_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_01_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_01_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_02_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_02_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_02_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_04_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_04_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_04_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_05_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_05_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_05_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_06_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_06_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_06_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_08_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_08_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_08_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_09_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_09_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_09_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_13_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_13_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_13_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_14_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_14_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_14_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_21_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_21_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_21_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_22_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_22_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_22_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_25_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_25_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_25_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_34_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_34_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_34_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_35_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_35_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_35_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_40_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_40_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_40_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_41_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_41_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_41_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_48_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_48_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_48_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_49_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_49_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_49_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_55_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_55_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_55_dnn_min_confidence_score_0.6.jpg?raw=true)
![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_61_dnn_min_confidence_score_0.2.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_61_dnn_min_confidence_score_0.4.jpg?raw=true) | ![alt](https://github.com/Salah-Akil/emotion-recognition/blob/main/face-detection-comparison-images/photo_61_dnn_min_confidence_score_0.6.jpg?raw=true)

The DNN model was by far the best face detection model for:

- Sideway faces
- Faces that contain occlusive wear such as glasses or masks
- Faces that are very dark in skin

It also performed good on faces that are far away from the camera, though not as good as MTCNN which reign supreme in that category.  
For the `min_confidence_score` the best value was `0.4` which gives a good balance between precision and low false positive detections.

### 3.4 Conclusion

As stated in the *Hypothesis* section, some models indeed had problems detecting people with very dark skin and faces that look sideways.
Between the three models used the best for general use is DNN by far. The second best is MTCNN. This proves how modern use of deep learning provides a better approach to computer vision tasks, such as face detection.

For the Emotion Detection script the best solution is to give the option to either use the DNN model or MTCNN one. This is because if the faces we want to analyse are located more than 10 meters from the camera, then the best model to use is MTCNN, if they are close to the camera then DNN is the most appropriate to use.
