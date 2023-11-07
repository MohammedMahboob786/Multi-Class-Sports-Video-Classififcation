# Multi-Class Sports Video Classififcation using ResNet-50 Architecture

This GitHub repository contains a Jupyter Notebook for video processing and classification using deep learning techniques. The code leverages various Python libraries and modules to achieve this task. Below is an overview of the code structure and its functionality:

## Table of Contents
- [Imports](#imports)
- [Data Collection](#data-collection)
- [Reading and Processing Images](#reading-and-processing-images)
- [Data Serialization](#data-serialization)
- [Data Preprocessing](#data-preprocessing)
- [Splitting Data](#splitting-data)
- [Data Augmentation](#data-augmentation)
- [Model Construction](#model-construction)
- [Model Training](#model-training)
- [Video Processing and Classification](#video-processing-and-classification)

---

## Imports
The code starts by importing various Python libraries and modules necessary for the task, including OpenCV (cv2), NumPy, pydot, graphviz, deque, matplotlib, pickle, scikit-learn, Keras, TensorFlow, and other utility modules for image processing and deep learning.

---

## Data Collection
- The `dataset_path` variable specifies the path to a dataset folder, which is expected to contain subfolders, each representing a class label for the images.

---

## Reading and Processing Images
- The code loops over the subfolders in the dataset folder.
- For each class subfolder, it reads and processes images (skipping some image file types) and stores them in a list (`data`). The class labels are also stored in a separate list (`LABELS`).

---

## Data Serialization
- The processed image data and class labels are serialized using the `pickle` module and saved to separate pickle files for future use. The serialized data is saved to files for easier data loading in the future.

---

## Data Preprocessing
- The serialized data is loaded back into the script, and the image data and class labels are converted to NumPy arrays.
- Label binarization is performed on the class labels using scikit-learn's `LabelBinarizer`.

---

## Splitting Data
- The dataset is split into training and testing sets using scikit-learn's `train_test_split` function.

---

## Data Augmentation
- Data augmentation is defined using Keras's `ImageDataGenerator`. Two augmentation generators are created: one for training data and one for validation/testing data.
- ImageNet mean subtraction is also configured for both data augmentation objects.

---

## Model Construction
- The code constructs a neural network model based on the ResNet-50 architecture using Keras.
- The head of the model is added on top of the pre-trained ResNet-50 base model, and certain layers in the "conv5_" group are set to be trainable for fine-tuning.

---

## Model Training
- The model is compiled with an Adam optimizer and categorical cross-entropy loss.
- The head of the network is trained on the training data using the data augmentation generators (`trainAug` and `valAug`) for a specified number of epochs (50 in this case).

---

## Video Processing and Classification
- The code loads a video file ("boxing.mp4").
- It processes the video frame by frame:
  - Predictions are made on each frame using the trained model.
  - A queue (`Queue`) stores the predictions over a fixed number of frames and calculates the mean prediction.
  - The class label with the highest average prediction is determined, and the video frame is annotated with the label.
  - The processed frames are written to an output video file.

---

