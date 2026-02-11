# *Fashion MNIST Classifier (Basic Neural Network Version)*

## Description

This project builds a basic fully connected neural network to classify clothing images from the Fashion MNIST dataset. It is a beginner-friendly 
implementation that focuses on understanding neural network fundamentals.

## Objective

To learn:<br>
Basics of neural networks<br>
Image normalization<br>
Label encoding<br>
Model training and evaluation<br>
Accuracy visualization<br>

## Features

Dataset loading<br>
Pixel normalization<br>
One-hot encoding labels<br>
Simple neural network model<br>
Training visualization graphs<br>
Model evaluation on test data<br>

## Tech Stack

Python<br>
TensorFlow / Keras<br>
NumPy<br>
Matplotlib<br>

## Dataset

Fashion MNIST dataset contains grayscale images of:<br>
Shirts<br>
Shoes<br>
Bags<br>
Dresses<br>
Coats<br>

## Usage

Open notebook.<br>
Run all cells.<br>
Model trains and displays performance graphs<br>
Final test accuracy is printed.<br>

## Model Architecture

Structure:<br>
Flatten layer<br>
Dense(128, ReLU)<br>
Dense(64, ReLU)<br>
Output layer (Softmax – 10 classes)<br>
Optimizer: SGD<br>
Loss: Categorical Crossentropy<br>

## Results

Outputs include:<br>
Training accuracy graph<br>
Validation accuracy graph<br>
Final test accuracy<br>

## Future Improvements

Replace DNN with CNN<br>
Add dropout<br>
Use Adam optimizer<br>
Add confusion matrix<br>

## Author

Surabhi<br>
Aspiring Data Scientist


# Fashion MNIST Classifier (Deep Neural Network Version)

## Description

This project implements a Deep Neural Network (DNN) to classify clothing images from the Fashion MNIST dataset. The model learns 
to identify 10 categories such as shirts, shoes, bags, and coats based on grayscale images. The notebook demonstrates a full deep 
learning workflow including preprocessing, training, validation splitting, callbacks, and performance monitoring.

## Objective

The main aim of this project is to learn:<br>
Multi-class image classification<br>
Neural network architecture design<br>
Model training optimization<br>
Validation techniques<br>
Performance monitoring using callbacks<br>

## Features

Dataset loading using Keras datasets<br>
Data normalization and reshaping<br>
One-hot encoding labels<br>
Deep neural network with multiple hidden layers<br>
TensorBoard logging<br>
Early stopping to prevent overfitting<br>
Training/validation split<br>

## Tech Stack

Python<br>
TensorFlow / Keras<br>
NumPy<br>
Matplotlib<br>
Scikit-learn<br>

## Dataset

Dataset used: Fashion MNIST<br>
It contains:<br>
60,000 training images<br>
10,000 testing images<br>
10 clothing classes<br>

## Usage

Open notebook.<br>
Run cells sequentially.<br>
Model trains automatically.<br>
Logs are generated for TensorBoard visualization.<br>

## Model Architecture

Network structure:<br>
Flatten layer<br>
Dense(512)<br>
Dense(256)<br>
Dense(128)<br>
Output layer (Softmax)<br>
Regularization:<br>
EarlyStopping callback<br>
TensorBoard monitoring<br>

## Results

The notebook outputs:<br>
Training accuracy<br>
Validation accuracy<br>
Training logs<br>
Final evaluation metrics<br>

## Future Improvements

Add dropout layers<br>
Try CNN architecture<br>
Hyperparameter tuning<br>
Model checkpoint saving<br>

## Author

Surabhi<br>
Aspiring Data Scientist
