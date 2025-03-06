# Predict-Fuel-Efficiency-using-TensorFlow
Machine Learning project to predict vehicle fuel efficiency using TensorFlow and Deep Neural Networks.
Project Overview

#Project overview
This project aims to predict the fuel efficiency (MPG - miles per gallon) of vehicles using Deep Neural Networks (DNNs) with TensorFlow and Keras. The model is trained on the Auto MPG dataset, which contains various features such as horsepower, weight, displacement, and acceleration.

The goal is to build an accurate regression model that can predict a vehicle’s fuel efficiency based on its specifications.
🧠 Machine Learning Framework: TensorFlow, Keras
🐍 Programming Language: Python
📊 Data Handling: Pandas, NumPy
🎨 Visualization: Matplotlib, Seaborn
🏎 Dataset: Auto MPG

Dataset Information 📊
The dataset used is the Auto MPG dataset, which contains the following attributes:

Column Name	Description
mpg	Miles per gallon (target variable)
cylinders	Number of cylinders in the engine
displacement	Engine displacement
horsepower	Horsepower of the vehicle
weight	Vehicle weight
acceleration	Acceleration (0-60 mph)
model year	Model year of the vehicle
origin	Country of origin (1 = USA, 2 = Europe, 3 = Japan)
car name	Name of the car (Categorical)

Data Preprocessing & Feature Engineering
Handling Missing Values - Filled missing values in the horsepower column with the median.
Feature Scaling - Used MinMaxScaler() to normalize numerical features to the range [0, 1].
Encoding Categorical Variables - Converted car names to numerical values using LabelEncoder().
Outlier Removal - Used Z-score method to remove extreme values (optional).
Target Variable Transformation - Applied log transformation to mpg to improve normality.

Exploratory Data Analysis (EDA)

MODEL ARCHITECHTURE
We built a Deep Neural Network (DNN) with the following structure:
Layer	Neurons	Activation
Input	Shape = (num_features,)	-
Dense	128	ReLU
Dense	64	ReLU
Dense	32	ReLU
Output	1	Linear

The model is compiled with:

Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Evaluation Metrics: Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE)

Results & Performance 📈
Loss vs Validation Loss
Observations:

The loss rapidly decreases and stabilizes around epoch 10, meaning the model is well-trained.
There is no significant overfittin

MAPE vs Validation MAPE
Observations:
The Mean Absolute Percentage Error (MAPE) stabilizes at a low value, meaning predictions are close to actual values



