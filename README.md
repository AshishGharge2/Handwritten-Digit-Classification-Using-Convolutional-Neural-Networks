MNIST Image Classification with CNN
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify hand-written digits from the MNIST dataset.

Project Overview
The MNIST dataset consists of 28x28 grayscale images of hand-written digits, with 60,000 images for training and 10,000 images for testing. This project uses a simple CNN model with two convolutional layers to perform image classification on the dataset.

Installation
To run this project, you need to have Python installed with the following packages:

TensorFlow
Keras
NumPy
Matplotlib
scikit-learn
You can install these dependencies using pip:

pip install tensorflow keras numpy matplotlib scikit-learn
Usage
Load and visualize data: The dataset is loaded from Keras and some sample images are displayed.
Preprocess data: Reshape the images and perform one-hot encoding on the labels.
Build and train the model: A simple CNN model is defined, compiled, and trained on the MNIST training set.
Evaluate the model: The model's accuracy is evaluated on the test set.



Results
The model achieves an accuracy score of approximately 97.8% on the test set after 3 epochs. 
Adjust the model architecture or increase the number of epochs to improve accuracy.
