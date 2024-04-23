# Machine-Learning-Task

Machine Learning Project
This repository contains Python code and resources for various machine learning tasks, including data cleaning, logistic regression, anomaly detection, image classification using CNN, and time series forecasting.

Files

Time Series Forecasting:

A code to perform time series forecasting.ipynb:
This script contains code for performing time series forecasting using techniques like ARIMA ,ARMA , SARIMA.
First important libraries are imported then data loading and processing is done.
The data is split into training and testing sets based on a specified date.
ARIMA (AutoRegressive Integrated Moving Average) model is built and fitted to the training data.
Prediction is made using the ARIMA model on the testing set.
The RMSE values for ARIMA, ARMA, and SARIMA models are calculated and printed.
The script plots the actual values, predicted values, and training/testing splits for the ARIMA, ARMA, and SARIMA models.
The RMSE values for each model are printed.


Data Preprocessing :

The script imports necessary libraries.
The dataset is loaded from a CSV file.
Data Type Conversionis done.
Some columns are converted to appropriate data types such as object, category, and datetime.
Missing values are checked using isnull().sum().
Outliers are handled using both Z-score and IQR methods.
Z-score normalization is performed to scale numerical features.
Data is scaled using both Z-score and Min-Max scaling techniques.
Categorical variables are label encoded using LabelEncoder.
The dataset is split into training and testing sets.
Median values for Y Train and Y Test variables are calculated.
Skewness and kurtosis of the original and transformed data are calculated.
Distribution plots are created to visualize the original and transformed data.


Image Classification using CNN:

This code implements a Convolutional Neural Network (CNN) model using the Keras library for image classification tasks. 

The code imports the necessary modules from Keras to build and train the CNN model.
An instance of the Sequential model is created, which allows adding layers sequentially.
The first Convolutional layer is added.
MaxPooling layer with a pool size of (2,2) is added to reduce spatial dimensions.
Another Convolutional layer is added with similar parameters as the first one.
After the convolutional and pooling layers, a Flatten() layer is added to convert the pooled feature maps into a single vector. This prepares the data for input into the fully connected layers.
The model is compiled using compile(), specifying the optimizer as 'adam', loss function as 'binary_crossentropy' and metrics to monitor during training as 'accuracy'.
ImageDataGenerator is used to generate augmented images for training data .
Augmentation techniques include rescaling, shear range, zoom range, and horizontal flip.


Anomaly Detection: Here, you'll find an implementation of an anomaly detection from scratch.

There is a functionality to read and write CSV files.
AnomalyDetector Class implements an anomaly detection algorithm based on Z-scores.
the cose Initializes the detector with a threshold value.
fit(X):its the detector to the provided data X by calculating mean and standard deviation for each feature.
detect_anomalies(X): Detects anomalies in the provided data X based on the calculated mean and standard deviation. Returns a boolean list indicating whether each sample is anomalous or not.
Code Converts the list of data to a numpy array X.
Anomaly Detection:
Initializes an AnomalyDetector object.
Fits the detector to the data.
Detects anomalies in the data.


Logistic Regression from scratch :

This script implements logistic regression from scratch. It includes functions for sigmoid function, log loss.

Data Preprocessing is done.
train_test_split, Splits the data into training and testing sets with a test size of 25%.
StandardScaler, Standardizes features by removing the mean and scaling to unit variance.
initialize_weights: Initializes the weights w to zeros and bias b to zero.
Sigmoid Function:
sigmoid: Implements the sigmoid activation function, which maps any real-valued number to the range [0, 1].
Log Loss Function:
logloss: Calculates the logistic loss or log loss, which is the loss function used in logistic regression.
Gradient Descent Functions:
gradient_dw: Calculates the gradient of the weight vector w.
gradient_db: Calculates the gradient of the bias term b.
Custom Logistic Regression Function:
custom_lr: Implements logistic regression training using gradient descent. It iterates through the training data for a specified number of epochs, updating weights and biases using gradients calculated by gradient_dw and gradient_db. It also calculates the training and testing log loss at each epoch.
Training:
alpha, eta, and epochs are hyperparameters for the learning rate, regularization parameter, and number of epochs, respectively.
w, b, log_loss_train, and log_loss_test are the outputs of the custom_lr function, representing the learned weights and biases, and the training and testing log loss values over epochs.
This code provides a basic implementation of logistic regression from scratch for binary classification tasks.











