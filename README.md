# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Data Collection & Preprocessing Import required libraries (pandas, numpy, matplotlib, sklearn). Load the dataset (student_scores.csv). Separate the independent variable (Hours) and dependent variable (Scores). Split the dataset into training and testing sets.

Step 2: Model Training Initialize the Linear Regression model. Train the model using the training dataset (x_train, y_train).

Step 3: Model Prediction & Visualization Predict scores for the test dataset. Plot the regression line with training data (gradient color for points). Plot the regression line with testing data (compare actual vs predicted values).

Step 4: Model Evaluation Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). Display evaluation results to assess model accuracy.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: CHARUKESH S
RegisterNumber:  21224230044
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv("student_scores.csv")
print("Displaying the first 5 Rows")
df.head()
```
```
print("displaying the Last 5 Rows")
df.tail()
```
```
x=df.iloc[:,:-1].values
x
```
```
y=df.iloc[:,-1].values
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=1/3,random_state=0)
```
```
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
```
```
y_test
```
```
print("Name: CHARUKESH S")
print("Reg.No: 212224230044")
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs Scores (Training dataset)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
print("Name: CHARUKESH S")
print("Reg.No: 212224230044")
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,regressor.predict(x_test),color="blue")
plt.title("Hours vs Scores (Test dataset)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
print("Name: CHARUKESH S")
print("Reg.No: 212224230044")

mse=mean_squared_error(y_test,y_pred)
print("\nMSE:",mse)
mae=mean_absolute_error(y_test,y_pred)
print("\nMAE:",mae)
rmse=np.sqrt(mse)
print("\nRMSE:",rmse)
```

## Output:
<img width="823" height="248" alt="image" src="https://github.com/user-attachments/assets/f44cdb4c-4e29-4029-97a5-c0adc6add094" />
<img width="991" height="243" alt="image" src="https://github.com/user-attachments/assets/29b514a5-067c-42a8-83f0-b02b084e62f1" />
<img width="786" height="547" alt="image" src="https://github.com/user-attachments/assets/edbd15e4-6bdf-461f-9ec4-0f57078c034c" />
<img width="847" height="63" alt="image" src="https://github.com/user-attachments/assets/ad62bdc4-2791-4a76-918e-0c51b5b8cc1e" />
<img width="847" height="68" alt="image" src="https://github.com/user-attachments/assets/6e8f1a41-5b88-4c3d-b706-4865aa5be1a6" />
<img width="682" height="45" alt="image" src="https://github.com/user-attachments/assets/a5c57ddc-9e6e-49c3-9b43-71c0b608cf92" />
<img width="900" height="642" alt="image" src="https://github.com/user-attachments/assets/72f07d95-53ef-4fe2-a816-809e42bac5d5" />
<img width="802" height="631" alt="image" src="https://github.com/user-attachments/assets/e6d6c4bc-fe40-49eb-b12b-23a0d66b069f" />
<img width="602" height="190" alt="image" src="https://github.com/user-attachments/assets/b3ff15e0-7bfe-4f7f-a857-332362d393c1" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
