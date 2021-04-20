# Datasets

In order to create a proper facial emotion recognition model we need a datasets that contains enough diversified data in order to be used for training the neural network.
Unfortunately there are not many free datasets available, I managed to download only two datasets:

* Extended Cohn-Kanade (CK+)
* Kaggle Face Expression Recognition Dataset

Both datasets contain `48x48` pixel images, in grayscale (1 channel).

## 1. Extended Cohn-Kanade (CK+)

### 1.1 Emotions

The CK+ dataset contains the 7 basic facial expressions, which are:

- *Anger*
- *Contempt*
- *Disgust*
- *Fear*
- *Happiness*
- *Sadness*
- *Surprise*

The problem is that it lacks a *neutral* class. This creates problems when performing the predictions since most people spend time with a neutral resting face rather than in on of the 7 basic facial expressions, resulting in many false positives.

### 1.2 Dataset Size

The problem with the CK+ dataset is its size. It only contains `327` images, which is very low in order to create a model capable of generalizing well. In fact models trained on this model performed very bad and were always having overfitting and sometimes underfitting problems.

Because of the low number of images I didn't create a test set, only the training and validations sets were created in order to maximize the training process.

### 1.3 Dataset Quality

The quality of the CK+ dataset is very good, it does not contain corrupted images or images that have incongruent face.

## 2. Kaggle Face Expression Recognition Dataset

### 2.1 Emotions

The Kaggle FAC dataset contains 6 emotions of the basic facial expressions, but it contain a *neutral* class:

- *Anger*
- *Disgust*
- *Fear*
- *Happiness*
- *Neutral*
- *Sadness*
- *Surprise*

### 2.2 Dataset Size

The Kaggle FAC dataset has high number of faces, for a total of `35.884` images. This images are not well distributed among the classes, for example the *happiness* folder in the training set contains `6493` images while the *disgust* folder only has `393` images. This leads to over confidence in the prediction. Overall the dataset contains enough images to train the model and achieve good results.

### 3.3 Dataset Quality

The quality of this dataset is not that good, since some of the images in the dataset are either completely black (corrupted) or contain text in it. It is a low number of unfitted files, but still it's very important to keep in mind.