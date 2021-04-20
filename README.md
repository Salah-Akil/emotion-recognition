# Emotion Recognition Study

## Abstract

This repository contains the study I went trough in order to create an emotion recognition neural network. I started from the very basics of machine learning and deep learning in order to understand how neural networks work in details and how to create different types of models for solving the emotion recognition task and how to train and fine-tuning them by using *transfer learning*.

The initial idea was to create a software capable of detecting faces in images or videos and perform emotion analysis on the detected faces. A further study was done on how modern and old face detection models work, a benchamrk test was created in order to pick the best model for face recognition, and only then I implemented the model in my software.

Emotion recognition is one of the harderst computer vision tasks to perform correctly, a 2020 [study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0231968#:~:text=The%20human%20recognition%20accuracy%20of,from%2048%25%20to%2062%25.) tested eight commercial classifiers, and compared their emotion recognition performance to that of human observers, revealing recognition advantage for human observers over automatic classification. Among the eight classifiers, there was considerable variance in recognition accuracy ranging from `48%` to `62%`.

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/emotion_performance_study.png?raw=true)

So my goal was to achieve at least `+50% accuracy` on a total of 7 emotion classes, which is better than pure chance standing at `14.28%`. Once the model is created and achieves the desired accuracy it can be deployed for real-life applications.

The following headings contain the all the needed knowledge to reproduce this results, form the theory about deep learning, how face detection works, required software and hardware, to how I created the models and results achieved.

## Index

### [Deep Learning](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#deep-learning)
#### [Introduction](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#1-introduction)
##### [What is Deep Learning](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/deep_learning.md#11-what-is-deep-learning)
