# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

STEP 1: Start the program.

STEP 2: Import the required library and read the dataframe.

STEP 3: Write a function computeCost to generate the cost function.

STEP 4: Perform iterations og gradient steps with learning rate.

STEP 5: Plot the Cost function using Gradient Descent and generate the required graph.

STEP 6: Stop the program.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SADHANA SHREE B
RegisterNumber: 212223230177
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions -y).reshape(-1,1)
        theta -= learning_rate *(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("/content/50_Startups.csv")
print(data.head())
X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
theta = linear_regression(X1_Scaled, Y1_Scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

```

## Output:
![Screenshot 2024-09-27 200320](https://github.com/user-attachments/assets/642997aa-63ef-4923-be72-8c53479283c8)
![Screenshot 2024-09-27 200334](https://github.com/user-attachments/assets/4d9fabc8-4a67-4574-b6d6-d4ba8c6f79ad)
![Screenshot 2024-09-27 200346](https://github.com/user-attachments/assets/08ce4b42-d08e-447e-8cb1-74c302114210)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
