# Emotion Recognition Study

## Abstract

This repository contains the study I went trough in order to create an emotion recognition neural network. I started from the very basics of machine learning and deep learning in order to understand how neural networks work in details and how to create different types of models for solving the emotion recognition task and how to train and fine-tuning them by using *transfer learning*.

The initial idea was to create a software capable of detecting faces in images or videos and perform emotion analysis on the detected faces. A further study was done on how modern and old face detection models work, a benchamrk test was created in order to pick the best model for face recognition, and only then I implemented the model in my software.

Emotion recognition is one of the harderst computer vision tasks to perform correctly, a 2020 [study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0231968#:~:text=The%20human%20recognition%20accuracy%20of,from%2048%25%20to%2062%25.) tested eight commercial classifiers, and compared their emotion recognition performance to that of human observers, revealing recognition advantage for human observers over automatic classification. Among the eight classifiers, there was considerable variance in recognition accuracy ranging from `48%` to `62%`.

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/emotion_performance_study.png?raw=true)

## Results

My goal was to achieve at least `+50% accuracy` on a total of 7 emotion classes, which is better than pure chance standing at `14.28%`. Once the model is created and achieves the desired accuracy it can be deployed for real-life applications.

After 32 iterations on different models created (on 2 different datasets), I picked the model with the highest classification accuracy, which was tested on 2867 images and got a global emotion average accuracy of `59%`, but based on the emotion the accuracy varies form `51%` for fear to `70%` for happiness.

<p align="center">
  <img width="461" height="390" src="https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/best_model_kaggle_v12_cmatrix.png?raw=true">
</p>

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/my_model_results.png?raw=true)

The following headings contain the all the needed knowledge to reproduce this results, form the theory about deep learning, how face detection works, required software and hardware, to how I created the models and results achieved.

## Index

### I. [Deep Learning](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#deep-learning)

<details>
  <summary>Click to expand</summary>
  
  #### 1. [Introduction](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#1-introduction)
  ##### 1.1 [What is Deep Learning](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#11-what-is-deep-learning)
  #### 2. [Artificial Neural Networks Architecture](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#2-artificial-neural-networks-architecture)
  ##### 2.1 [How ANNs work?](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#21-how-anns-work)
  ##### 2.2 [What Layers do?](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#22-what-layers-do)
  ##### 2.3 [What are Activation Functions?](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#23-what-are-activation-functions)
  ###### 2.3.1 [Sigmoid Activation Function](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#sigmoid-activation-function)
  ###### 2.3.2 [ReLU Activation Function](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#relu-activation-function)
  ###### 2.3.3 [Softmax Activation Function](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#softmax-activation-function)
  #### 3. [Training an Artificial Neural Network](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#3-training-an-artificial-neural-network)
  ##### 3.1 [What it means to train a network?](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#31-what-it-means-to-train-a-network)
  ##### 3.2 [Stochastic Gradient Descent (SGD)](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#32-stochastic-gradient-descent-sgd)
  ##### 3.3 [Loss Function](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#33-loss-function)
  ##### 3.4 [(MSE) Mean Square Error](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#34-mse-mean-square-error)
  ##### 3.5 [Gradient of the Loss Function](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#35-gradient-of-the-loss-function)
  ##### 3.6 [How the network learn](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#36-how-the-network-learn)
  ##### 3.7 [Weights Update](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#37-weights-update)
  ##### 3.8 [Learning Rate](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#38-learning-rate)
  #### 4 [Datasets](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#4-datasets)
  ##### 4.1 [Training Set](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#41-training-set)
  ##### 4.2 [Validation Set](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#42-validation-set)
  ##### 4.3 [Test Set](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#43-test-set)
  ##### 4.4 [Overfitting](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#44-overfitting)
  ##### 4.5 [Underfitting](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#45-underfitting)
  #### 5 [Convolutional Neural Networks](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#5-convolutional-neural-networks)
  ##### 5.1 [Convolutional Layers](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#51-convolutional-layers)
  ##### 5.2 [Patterns](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#52-patterns)
  ##### 5.3 [Kernel](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#53-kernel)
  ##### 5.4 [Feature (Patter) Detection](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#54-feature-patter-detection)
  ##### 5.5 [Max Pooling](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#55-max-pooling)
  ##### 5.6 [Batch Size](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#56-batch-size)
  ##### 5.7 [Batch Normalization](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#57-batch-normalization)
  ##### 5.8 [Fine Tuning](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#58-fine-tuning)
  
</details>

### II. [Haar Cascades](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/haar_face_detection.md#haar-cascades)

<details>
  <summary>Click to expand</summary>

  #### 1. [Grayscale vs Colored Images](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/haar_face_detection.md#grayscale-vs-colored-images)
  ##### 1.1 [Grayscale](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/haar_face_detection.md#grayscale)
  ##### 1.2 [Colored Images](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/haar_face_detection.md#colored-images)
  #### 2. [How it works](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/haar_face_detection.md#how-it-works)
  #### 3. [Haar-Features](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/haar_face_detection.md#haar-features)
  #### 4. [Algorithm](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/haar_face_detection.md#algorithm)
  #### 5. [Integral Image](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/haar_face_detection.md#integral-image)
  #### 6. [AdaBoost](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/haar_face_detection.md#adaboost)
  #### 7. [Cascade](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/haar_face_detection.md#cascade)
  #### 8. [Haar Code Implementation](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/haar_face_detection.md#haar-code-implementation)
  ##### 8.1 [Installation & Dependencies](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/haar_face_detection.md#installation--dependencies)
  ##### 8.2 [Face Detection Script](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/haar_face_detection.md#face-detection-script)

  
</details>

### III. [MTCNN](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/mtcnn_face_detection.md#mtcnn)

<details>
  <summary>Click to expand</summary>
  
  #### 1. [MTCNN Stages](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/mtcnn_face_detection.md#1-mtcnn-stages)
  ##### 1.1 [Stage 1 (P-Net)](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/mtcnn_face_detection.md#11-stage-1-p-net)
  ##### 1.2 [Stage 2 (R-Net)](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/mtcnn_face_detection.md#12-stage-2-r-net)
  ##### 1.3 [Stage 3 (O-Net)](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/mtcnn_face_detection.md#13-stage-3-o-net)
  #### 2. [MTCNN Code Implementation](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/mtcnn_face_detection.md#2-mtcnn-code-implementation)
  ##### 2.1 [Installation & Dependencies](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/mtcnn_face_detection.md#21-installation--dependencies)
  ##### 2.2 [Face Detection Script](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/mtcnn_face_detection.md#22-face-detection-script)

  
</details>

### IV. [DNN Face Detector in OpenCV](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/dnn_face_detection.md#dnn-face-detector-in-opencv)

<details>
  <summary>Click to expand</summary>
  
  #### 1. [Single Shot Multibox Detector](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/dnn_face_detection.md#single-shot-multibox-detector)
  ##### 1.1 [Model](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/dnn_face_detection.md#model)
  ##### 1.2 [NMS](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/dnn_face_detection.md#nms)
  ##### 1.3 [Stage 3 (O-Net)](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/mtcnn_face_detection.md#13-stage-3-o-net)
  #### 2. [DNN Code Implementation](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/dnn_face_detection.md#dnn-code-implementation)
  ##### 2.1 [Installation & Dependencies](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/dnn_face_detection.md#installation--dependencies)
  ##### 2.2 [Face Detection Script](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/dnn_face_detection.md#face-detection-script)

  
</details>

### V. [Face Detection Models Comparison](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/fd_models_comparison.md#face-detection-models-comparison)

<details>
  <summary>Click to expand</summary>
  
  #### 1. [Dataset](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/fd_models_comparison.md#1-dataset)
  ##### 1.1 [Hypothesis](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/fd_models_comparison.md#11-hypothesis)
  #### 2. [Comparison](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/fd_models_comparison.md#2-comparison)
  ##### 2.1 [Code](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/fd_models_comparison.md#22-code)
  #### 3. [Results](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/fd_models_comparison.md#3-results)
  ##### 3.1 [HAAR](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/fd_models_comparison.md#31-haar)
  ##### 3.2 [MTCNN](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/fd_models_comparison.md#32-mtcnn)
  ##### 3.3 [DNN](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/fd_models_comparison.md#33-dnn)
  ##### 3.4 [Conclusion](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/fd_models_comparison.md#34-conclusion)

  
</details>

### VI. [Facial Expressions](https://github.com/Salah-Akil/emotion-recognition/)

<details>
  <summary>Click to expand</summary>
  
  #### 1. [To-Add](https://github.com/Salah-Akil/emotion-recognition/)

  
</details>

### VII. [Datasets](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/datasets.md#datasets)

<details>
  <summary>Click to expand</summary>
  
  #### 1. [Extended Cohn-Kanade (CK+)](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/datasets.md#1-extended-cohn-kanade-ck)
  ##### 1.1 [Emotions](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/datasets.md#11-emotions)
  ##### 1.2 [Dataset Size](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/datasets.md#12-dataset-size)
  ##### 1.3 [Dataset Quality](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/datasets.md#13-dataset-quality)
  #### 2. [Kaggle Face Expression Recognition Dataset](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/datasets.md#2-kaggle-face-expression-recognition-dataset)
  ##### 2.1 [Emotions](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/datasets.md#21-emotions)
  ##### 2.2 [Dataset Size](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/datasets.md#22-dataset-size)
  ##### 2.3 [Dataset Quality](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/datasets.md#33-dataset-quality)

  
</details>

### VIII. [Neural Networks Development](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/neural_networks_development.md#neural-networks-development)

<details>
  <summary>Click to expand</summary>
  
  #### 1. [Neural Networks with CK+ Dataset](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/neural_networks_development.md#1-neural-networks-with-ck-dataset)
  ##### 1.1 [gen-01](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_01/ck_network_v1.ipynb)
  ##### 1.2 [gen-02](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_02/ck_network_v2.ipynb)
  ##### 1.3 [gen-03](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_03/ck_network_v3.ipynb)
  ##### 1.4 [gen-04](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_04/ck_network_v4.ipynb)
  ##### 1.5 [gen-05](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_05/ck_network_v5.ipynb)
  ##### 1.6 [gen-06](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_06/ck_network_v6.ipynb)
  ##### 1.7 [gen-07](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_07/ck_network_v7.ipynb)
  ##### 1.8 [gen-08](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_08/ck_network_v8.ipynb)
  ##### 1.9 [gen-09](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_09/ck_network_v9.ipynb)
  ##### 1.10 [gen-10](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_10/ck_network_v10.ipynb)
  ##### 1.11 [gen-11](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_11/ck_network_v11.ipynb)
  ##### 1.12 [gen-12](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_12/ck_network_v12.ipynb)
  ##### 1.13 [gen-13](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_13/ck_network_v13.ipynb)
  ##### 1.14 [gen-14](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_14/ck_network_v14.ipynb)
  ##### 1.15 [gen-15](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_15/ck_network_v15.ipynb)
  ##### 1.16 [gen-16](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_16/ck_network_v16.ipynb)
  ##### 1.17 [gen-17](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_17/ck_network_v17.ipynb)
  ##### 1.18 [gen-18](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_18/ck_network_v18.ipynb)
  #### 2. [Neural Networks with Kaggle FAC Dataset](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/neural_networks_development.md#1-neural-networks-with-kaggle-fac-dataset)
  ##### 2.1 [gen-01](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_01/kaggle_network_v1.ipynb)
  ##### 2.2 [gen-02](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_02/kaggle_network_v2.ipynb)
  ##### 2.3 [gen-03](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_03/kaggle_network_v3.ipynb)
  ##### 2.4 [gen-04](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_04/kaggle_network_v4.ipynb)
  ##### 2.5 [gen-05](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_05/kaggle_network_v5.ipynb)
  ##### 2.6 [gen-06](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_06/kaggle_network_v6.ipynb)
  ##### 2.7 [gen-07](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_07/kaggle_network_v7.ipynb)
  ##### 2.8 [gen-08](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_08/kaggle_network_v8.ipynb)
  ##### 2.9 [gen-09](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_09/kaggle_network_v9.ipynb)
  ##### 2.10 [gen-10](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_10/kaggle_network_v10.ipynb)
  ##### 2.11 [gen-11](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_11/kaggle_network_v11.ipynb)
  ##### 2.12 [gen-12](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_12/kaggle_network_v12.ipynb)
  ##### 2.13 [gen-13](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_13/kaggle_network_v13.ipynb)
  ##### 2.14 [gen-14](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_14/kaggle_network_v14.ipynb)

  
</details>

### IX. [Results](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/neural_networks_results.md#neural-networks-results)

<details>
  <summary>Click to expand</summary>
  
  #### 1. [Neural Networks with CK+ Dataset](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/neural_networks_development.md#1-neural-networks-with-ck-dataset)
  ##### 1.1 [gen-01](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_01/test_model_v1.ipynb)
  ##### 1.2 [gen-02](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_02/test_model_v2.ipynb)
  ##### 1.3 [gen-03](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_03/test_model_v3.ipynb)
  ##### 1.4 [gen-04](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_04/test_model_v4.ipynb)
  ##### 1.5 [gen-05](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_05/test_model_v5.ipynb)
  ##### 1.6 [gen-06](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_06/test_model_v6.ipynb)
  ##### 1.7 [gen-07](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_07/test_model_v7.ipynb)
  ##### 1.8 [gen-08](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_08/test_model_v8.ipynb)
  ##### 1.9 [gen-09](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_09/test_model_v9.ipynb)
  ##### 1.10 [gen-10](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_10/test_model_v10.ipynb)
  ##### 1.11 [gen-11](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_11/test_model_v11.ipynb)
  ##### 1.12 [gen-12](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_12/test_model_v12.ipynb)
  ##### 1.13 [gen-13](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_13/test_model_v13.ipynb)
  ##### 1.14 [gen-14](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_14/test_model_v14.ipynb)
  ##### 1.15 [gen-15](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_15/test_model_v15.ipynb)
  ##### 1.16 [gen-16](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_16/test_model_v16.ipynb)
  ##### 1.17 [gen-17](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_17/test_model_v17.ipynb)
  ##### 1.18 [gen-18](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/CK%2B/gen_18/test_model_v18.ipynb)
  #### 2. [Neural Networks with Kaggle FAC Dataset](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/neural_networks_development.md#1-neural-networks-with-kaggle-fac-dataset)
  ##### 2.1 [gen-01](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_01/test_model_v1.ipynb)
  ##### 2.2 [gen-02](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_02/test_model_v2.ipynb)
  ##### 2.3 [gen-03](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_03/test_model_v3.ipynb)
  ##### 2.4 [gen-04](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_04/test_model_v4.ipynb)
  ##### 2.5 [gen-05](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_05/test_model_v5.ipynb)
  ##### 2.6 [gen-06](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_06/test_model_v6.ipynb)
  ##### 2.7 [gen-07](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_07/test_model_v7.ipynb)
  ##### 2.8 [gen-08](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_08/test_model_v8.ipynb)
  ##### 2.9 [gen-09](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_09/test_model_v9.ipynb)
  ##### 2.10 [gen-10](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_10/test_model_v10.ipynb)
  ##### 2.11 [gen-11](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_11/test_model_v11.ipynb)
  ##### 2.12 [gen-12](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_12/test_model_v12.ipynb)
  ##### 2.13 [gen-13](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_13/test_model_v13.ipynb)
  ##### 2.14 [gen-14](https://github.com/Salah-Akil/emotion-recognition/blob/main/archive/Kaggle/gen_14/test_model_v14.ipynb)

  
</details>

### X. [Requirements](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/requirements.md#requirements)

### XI. [Hardware](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/hardware.md#hardware)
