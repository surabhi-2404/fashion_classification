Fashion MNIST Classifier (Basic Neural Network Version)

Description

This project builds a basic fully connected neural network to classify clothing images from the Fashion MNIST dataset. It is a beginner-friendly 
implementation that focuses on understanding neural network fundamentals.

Objective

To learn:
Basics of neural networks
Image normalization
Label encoding
Model training and evaluation
Accuracy visualization

Features

Dataset loading
Pixel normalization
One-hot encoding labels
Simple neural network model
Training visualization graphs
Model evaluation on test data

Tech Stack

Python
TensorFlow / Keras
NumPy
Matplotlib

Dataset

Fashion MNIST dataset contains grayscale images of:
Shirts
Shoes
Bags
Dresses
Coats
etc.

Installation
pip install tensorflow matplotlib numpy

Run notebook:

jupyter notebook

Usage

Open notebook.
Run all cells.
Model trains and displays performance graphs
Final test accuracy is printed.

Model Architecture

Structure:
Flatten layer
Dense(128, ReLU)
Dense(64, ReLU)
Output layer (Softmax – 10 classes)
Optimizer: SGD
Loss: Categorical Crossentropy

Results

Outputs include:
Training accuracy graph
Validation accuracy graph
Final test accuracy

Future Improvements

Replace DNN with CNN
Add dropout
Use Adam optimizer
Add confusion matrix

Author

Surabhi
Aspiring Data Scientist


Fashion MNIST Classifier (Deep Neural Network Version)

📖 Description

This project implements a Deep Neural Network (DNN) to classify clothing images from the Fashion MNIST dataset. The model learns 
to identify 10 categories such as shirts, shoes, bags, and coats based on grayscale images. The notebook demonstrates a full deep 
learning workflow including preprocessing, training, validation splitting, callbacks, and performance monitoring.

🎯 Objective

The main aim of this project is to learn:
Multi-class image classification
Neural network architecture design
Model training optimization
Validation techniques
Performance monitoring using callbacks

🚀 Features

Dataset loading using Keras datasets
Data normalization and reshaping
One-hot encoding labels
Deep neural network with multiple hidden layers
TensorBoard logging
Early stopping to prevent overfitting
Training/validation split

🛠️ Tech Stack

Python
TensorFlow / Keras
NumPy
Matplotlib
Scikit-learn

📂 Dataset

Dataset used: Fashion MNIST
It contains:
60,000 training images
10,000 testing images
10 clothing classes

⚙️ Installation

pip install tensorflow matplotlib numpy scikit-learn

Run notebook:

jupyter notebook

▶️ Usage

Open notebook.
Run cells sequentially.
Model trains automatically.
Logs are generated for TensorBoard visualization.

🧠 Model Architecture

Network structure:
Flatten layer
Dense(512)
Dense(256)
Dense(128)
Output layer (Softmax)
Regularization:
EarlyStopping callback
TensorBoard monitoring

📊 Results

The notebook outputs:
Training accuracy
Validation accuracy
Training logs
Final evaluation metrics

📈 Future Improvements

Add dropout layers
Try CNN architecture
Hyperparameter tuning
Model checkpoint saving

👩‍💻 Author

Surabhi
Deep Learning Learner
