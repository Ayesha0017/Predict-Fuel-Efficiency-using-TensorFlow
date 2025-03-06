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
#Shows the correlation between numerical features in the dataset.
plt.figure(figsize=(8, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
#horsepower and weight show a negative relationship with mpg.

#pairplot of numerical features
plt.figure(figsize=(8, 4))
sns.pairplot(df[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']])
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()

#histogram of target variable. The frequency of MPG values is shown in the histogram
plt.figure(figsize=(8, 4))
sns.histplot(df['mpg'], bins=20, kde=True)
plt.title('Distribution of MPG')
plt.xlabel('MPG')
plt.ylabel('Frequency')
plt.show()

#scatter plot of MPG vs Horsepower. The relationship between MPG and Horsepower is shown in the scatter plot
plt.figure(figsize=(8, 4))
sns.scatterplot(x='horsepower', y='mpg', data=df)
plt.title('MPG vs Horsepower')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.show()


#scatter plot of MPG vs Weight. The relationship between MPG and Weight is shown in the scatter plot
plt.figure(figsize=(8, 4))
sns.scatterplot(x='weight', y='mpg', data=df)
plt.title('MPG vs Weight')
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

#box plot of numerical features. The distribution of numerical features is shown in the box plot
plt.figure(figsize=(8, 4))
sns.boxplot(data=df[numeric_cols])
plt.title('Box Plot of Numerical Features')
plt.show()

#count plot of origin. The count of cars by origin is shown in the count plot
plt.figure(figsize=(8, 4))
sns.countplot(x='origin', data=df)
plt.title('Count of Cars by Origin')
plt.show()

MODEL ARCHITECHTURE
We built a Deep Neural Network (DNN) with the following structure:
Layer	Neurons	Activation
Input	Shape = (num_features,)	-
Dense	128	ReLU
Dense	64	ReLU
Dense	32	ReLU
Output	1	Linear

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

The model is compiled with:

Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Evaluation Metrics: Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE)

TRAINING THE MODEL
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_split=0.3, epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping], shuffle=True)

Early stopping is used to prevent overfitting.
The model was trained for 100 epochs, but training stopped early when validation loss stopped improving.

Results & Performance 📈
Loss vs Validation Loss
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

Observations:

The loss rapidly decreases and stabilizes around epoch 10, meaning the model is well-trained.
There is no significant overfittin

MAPE vs Validation MAPE

plt.plot(history.history['mape'], label='MAPE')
plt.plot(history.history['val_mape'], label='Validation MAPE')
plt.title('MAPE vs Validation MAPE')
plt.xlabel('Epochs')
plt.ylabel('MAPE')
plt.legend()
plt.show()




