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

Anomaly Detection: Here, you'll find an implementation of an anomaly detection from scratch. The script identifies anomalies in a dataset based on deviations from a normal distribution.


Logistic Regression from scratch :
This script implements logistic regression from scratch. It includes functions for sigmoid function, log loss.
Data Generation:
make_classification: This function generates a random n-class classification problem. In this case, it generates 50,000 samples with 15 features, where 10 features are informative and 5 are redundant. The classes are binary, and the class separation is set to 0.7.
Data Preprocessing:
train_test_split: Splits the data into training and testing sets with a test size of 25%.
StandardScaler: Standardizes features by removing the mean and scaling to unit variance.
Initialization:
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











