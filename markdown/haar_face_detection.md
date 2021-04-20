# Haar Cascades

As we can deduce face detection is the task of finding (detecting) and recognizing faces in images and videos. The standard idea nowadays is to use deep learning, which is true, but in reality the approach that most are using is not deep learning, but the **_Viola-Jones algorithm_**, also known as **_Haar Cascades_**.

In 2001 Paul Viola and Michael Jones published a paper called "Rapid Object Detection using Boosted Cascade of Simple Features".

Despite being more than 20 years old, this algorithm is still widely used and quite powerful.

## 1. Grayscale vs Colored Images

The Viola-Jones algorithm only works on grayscale images since they contain less data than colored ones. In fact many computer vision task use grayscale images. Let's explain why.

### 1.1 Grayscale

A grayscale image just need information about how bright a particular pixel is, basically it's intensity. The higher the value, the greater the intensity. A display normally supports 256 different shades of gray, so for a single grayscale image we just need a single byte to represent each pixel since:

1 Byte = 8 bits &rarr; 256 possible values

![alt text](images\camerino_grayscale.png "Grayscale Image Representation")

So in the memory a grayscale image is represented by a two dimensional array of bytes. The dimensions of this array (called also "channel") are equal to the height and width of the image. So, a grayscale image has only one channel. And this channel represents the intensity of gray.

### 1.2 Colored Images

Colored images work differently and require much more data to be stored than a simple grayscale one. We are not dealing anymore with just about what shade, we start dealing with what shade of which "color".

A normal JPEG image supports more than 16 million different colors. In order to be able to store represent all these colors we use 3 bytes for each pixel:

3 Bytes &rarr; 24 bits &rarr; 16,777,216 possible values

Since an 8-bit image allows us 256 shaded of grey per channel, for a 24-bit we can use 3 channels (R,G,B).

<!-- ![alt text](images\colorspace_example.png "RGB") -->

<p align="center">
  <img width="330" height="240" src="images\colorspace_example.png">
</p>

We use red, green and blue since being the primary colors can be mixed together to form any other color. In fact have 256 different shades of red, green and blue (we have seen that 1 byte can store a value from 0 to 255). So by mixing these colors in different proportions, we get the desired color.

![alt text](images\camerino_rgb.png "RGB Image Representation")


As we can see in the image above, the color (Blue Marine) in the pixel is represented by an RGB with values set to:

- Red &rarr; 96
- Green &rarr; 133
- Blue &rarr; 160

## 2. How it works

The Viola-Jones algorithm was designed to detect frontal faces in images, rather than faces looking upwards, downwards or sideways. In fact if we feed as input an image with a face looking sideways to the `face_detection.py` script (we will talk about it later) we will see a decrease in detection accuracy.

What the algorithm does is to outline a box and move it every time a step to the right until it reaches the end and repeat this process for each tile in the picture. The goal is to search for a face features such as eyes, eyebrows, lips or a nose within this box every time it moves, and in order to detect these features we use *Haar-Features*.

## 3. Haar-Features

Haar-Features are named after Haar Wavelets. A haar-wavelet is a sequence of rescaled square-shaped functions (very similar to Fourier-analysis) that were proposed by a hungarian mathematician named Alfred Haar in 1909.

(Haar features are very similar to convolutional kernels on how they work)

The haar features proposed by Viola and Jones are boxes composed of a light (white) part and a dark (black) side, and thanks to this contrast it's possible to determine we have found a face feature or not. This is because if we take a grayscale scale image of a face, some part will be darker than other part in the face, for example the eyebrows, lips or eyes are darker than the forehead or cheeks. Sometimes the middle section may be lighter than the adjacent boxes, in which case it can be interpreted as a nose.

![alt text](images\grayscale_eyebrow_forehead.png "Grayscale Eyebrow Feature")

For example in the image above we can see that the eyebrow section is darker than the forehead, making the eyebrow a facial feature that we can detect with a haar feature.

The Viola-Jones paper identifies 3 types of Haar features:

- Edge features
- Line features
- Four-rectangle features

But the ones needed for face detection are the **edge features** and **line features**.

![alt text](images\haar_features.png "Haar Features")

Since we can use them to represent the most relevant features of a face.

<p align="center">
    <img width="450" height="474" src="images\face_with_haar_boxes.png">
</p>

## 4. Algorithm

As we explained, a facial features like an eyebrow is composed of a dark part (eyebrow) and a light part (the skin above the eyebrow). In a grayscale image the eyebrow will be represented by darker (high value) pixel and the skin part by lighter (low value) pixels. We know that this pixel values are in the range of 0 to 255, but for simplicity we will represented each pixel with a value from 0.00 to 1.00.

The Viola-Jones Algorithm defines an ideal scenario for a haar feature were the darker pixels are all set to 1 and the lighter pixels to 0, creating two zones (for the feature below an upper light zone and a darker bottom zone).

<p align="center">
    <img width="618" height="342" src="images\haar_pixel_ideal.png">
</p>

But the facial features detected in a grayscale image will have more realistic values, for example:

<p align="center">
    <img width="618" height="342" src="images\haar_pixel_real.png">
</p>

For both the ideal and real features we have to calculate the delta Δ, which represents the difference between the average sum of all pixel values of the bottom darker zone and upper lighter zone (always darker zone - lighter zone).

$$
Δ = dark - light =  \frac{1}{n}\sum_{dark}^{n} I(x) - \frac{1}{n}\sum_{light}^{n}I(x)
$$

*I(x)* &rarr; Pixel intensity of a given pixel x.

The results are:

- Δ for the ideal Haar feature &rarr; 1 - 0 = 1
- Δ for the real feature above &rarr; 0.77 - 0.25 = 0.52

In other words the closer the Δ is to 1 the more likely we have found a haar feature. In real life we will almost never get a Δ = 1, but Δ = 0.52 is a good real value.

By repeating this operation with the other haar edge and line features across the entire image we can detect facial features such as the nose, eyes, lips, eyebrows etc.

## 5. Integral Image

The process of calculating the average sum of all the pixel values in the haar features (edge and line of different kernel sizes) can be time consuming, for example a single haar feature can be composed of hundreds of pixels itself, while the entire images is composed of ten of thousands of pixels, the time complexity running these operation on the entire image is **O(N<sup>2</sup>)**.

![alt text](images\haar_different_kernel_size.png "Haar feature with different kernel sizes")
(***Put image of haar feature with different kernel sizes***)

To solve this problem, we can use *Integral Image* approach to achieve **O(1)** running time.
Integral Image is one of the most important tools used to accelerate features computation in many object detection applications. They are also known as Summed Area Tables and they were proposed in 1984 by Frank Crow.

<p align="center">
    <img width="640" height="480" src="images\time_complexity.png">
</p>

In order to achieve this we convert our original image into an integral image, where a given pixel *(x,y)* in the integral image is the sum of all the pixels to the left and above of the pixel *(x,y)*, including *i(x,y)*, according to equation (1), where *i(x,y)* is the value of the pixel at the position *(x,y)*.

$$
II (x,y) = \sum_{x' \leq x,y' \leq y} i(x',y'
$$

For example in the images below we can see that the highlighted pixel in the integral image is the sum of all the highlighted pixels in the original image.

![alt text](images\integral_image_14.png "Integral Image")

![alt text](images\integral_image_161.png "Integral Image")

We can also use recursion to calculate the computation of integral image *II(x,y)* as we can observe in the following equation (2):

$$
II (x,y) = i(x,y) - II(x-1,y-1) + II(x,y-1) + II(x-1,y)
$$

Were *i(x,y)* is the pixel value in the original image.

![alt text](images\integral_image_recursion.png "Integral Image")

For example the the pixel (x2,y1) in the integral image above can be calculated as:

$$II (x2,y1) = i(x2,y1) - II(x2-1,y1-1) + II(x2,y1-1) + II(x2-1,y1)$$
$$II (x2,y1) = i(x2,y1) - II(x1,y0) + II(x2,y0) + II(x1,y1)$$
$$II (x2,y1) = 4 - 2 + 5 + 7$$
$$II (x2,y1) = 14$$

Let's assume now we have to calculate the sum of the pixel intesity of a particular region in the original image, instead of calculating the all the pixels (which has a running time of O(N<sup>2</sup>)), we can simply manipulate the values in the integral image and get the same result.

Assuming we declare:

- RS(o) as the sum of all the pixel values in a particular region in the original image
- TL(x,y) as the top-left pixel of the same region but in the integral image
- TR(x,y) as the top-right pixel of the same region but in the integral image
- BL(x,y) as the bottom-left pixel of the same region but in the integral image
- BR(x,y) as the bottom-right pixel of the same region but in the integral image

![alt text](images\region_sum.png "Integral Image")

Then we can define:

$$ RS(o) = BR(x,y) - TR(x,y-1) + TL(x-1,y-1) - BL(x-1,y) $$

Which we can use to calculate the sum of all the pixels in the original image above as:

$$ RS(o) = BR(x5,y6) - TR(x5,y2-1) + TL(x3-1,y2-1) - BL(x3-1,y6) $$
$$ RS(o) = BR(x5,y6) - TR(x5,y1) + TL(x2,y1) - BL(x2,y6) $$
$$ RS(o) = 207 - 33 + 14 - 87 $$
$$ RS(o) = 101 $$

As we can see, instead of considering all the values in the haar feature region (a region might contain hundreds or thousands of pixels) and computing the sum, we just need 4 single pixel values from the integral image to get the same result, achieving *O(1)* running time.

## 6. AdaBoost

On the original paper Jones and Viola proposed 180.000 initial features. Most of these initial features were not suitable or were irrelevant to facial features, so they proposed the use of a *feature selection* technique in order to select only the relevant features needed for face features detection. The decision was to use a boosting technique called AdaBoost. We are not going to dive in how [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost) works since is not relevant to our project.
What we need to know is that these 180.000 initial features were reduced to only 6.000 essential features needed for face features detection.


## 7. Cascade

Having to run 6.000 features within a 24 by 24 windows on an image is still a time consuming task. To reduce this computational time Viola and Jones proposed the use of another technique named "*Attentional Cascade*", deducing that not all the features need to run on every window. If a particular feature fails on a window then we can deduce that the facial feature is not present, consequently moving to the next window and repeating the process.

These features are applied on the images in stages, with the the initial stages containing simpler and less features compared to the features in the later stages. In the original paper Viol and Jones proposed 38 stages in total. Here are the number of features contained in the first 5 stages:

| Stage number | Number of Features |
|:------------:|:------------------:|
|       1      |          1         |
|       2      |         10         |
|       3      |         25         |
|       4      |         25         |
|       5      |         50         |
|      ...     |         ...        |


These initial stages containing simpler and fewer features eliminate most of the non essential windows which don't contain any facial feature, enabling the detection to run on real-time on modern hardware architecture.

## 8. Haar Code Implementation

Let's now see how we can implement the haar face detection in our code.

### 8.1 Installation & Dependencies

Haar Cascades require OpenCV (for reading the cascade file) and Numpy packages, which can be installed via `pip` command:

```bash
pip install opencv-python
pip install numpy
```

Or in case we are running a Conda environment we can use `conda` command:

```bash
conda install -c conda-forge opencv
conda install -c anaconda numpy
```

Once installed we have to download the haar cascade itself in XML format, which can be found in the original OpenCV Github repository [https://github.com/opencv/opencv/tree/master/data/haarcascades]. The one we need for face detection is `haarcascade_frontalface_default.xml`.

### 8.2 Face Detection Script

We are now going to python function that takes two parameters:

- **img_path** &rarr; the path of the image we want to perform the face detection on
- **scale_factor** &rarr; the scaleFactor value (which specifies how much the image size is reduced at each image scale, and we set it to 1.10 by default)

The function will take the image path, read the image, perform the face detection task, and save as output a copy of the original image with possible bounding boxes that contain the faces detected.

```python
import cv2
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
```

We will later use this function to perform face detection on a set of images and compare them with the other face detection methods.
