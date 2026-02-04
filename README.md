# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the placement dataset and display a few records to understand its structure.

2. Convert categorical attributes (gender, work experience, specialization, etc.) into numerical form and remove irrelevant attributes such as salary.

3. Select student academic and profile attributes as input features and set placement status as the target variable.

4. Divide the dataset into training data and testing data to train and evaluate the model.

5. Fit the logistic regression classifier using the training dataset.

6. Predict placement status for testing data and generate confusion matrix, accuracy, precision, recall, and specificity.

7. Accept input details of a new student and use the trained model to predict whether the student will be placed or not.

## Program:
```
# Logistic Regression to Predict Placement Status of Students
# Developed By: Jeensfer Jo
# Register Number : 212225240058

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
df = pd.read_csv("Placement_Data.csv")

print("Dataset Preview:")
print(df.head())

# --------------------------------------------------
# 2. Data Preprocessing
# --------------------------------------------------
# Drop irrelevant columns
df.drop(columns=['sl_no', 'salary'], inplace=True)

# Separate features and target
X = df.drop('status', axis=1)
y = df['status']

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Encode target variable (Placed = 1, Not Placed = 0)
y = y.map({'Not Placed': 0, 'Placed': 1})

print("\nAfter Encoding:")
print(X.head())

# --------------------------------------------------
# 3. Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# 4. Train Logistic Regression Model
# --------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --------------------------------------------------
# 5. Model Evaluation
# --------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(cm)

# --------------------------------------------------
# 6. Predict Placement for a New Student
# --------------------------------------------------
print("\nEnter new student details:")

gender = input("Gender (M/F): ")
ssc_p = float(input("SSC Percentage: "))
ssc_b = input("SSC Board (Central/Others): ")
hsc_p = float(input("HSC Percentage: "))
hsc_b = input("HSC Board (Central/Others): ")
hsc_s = input("HSC Stream (Science/Commerce/Arts): ")
degree_p = float(input("Degree Percentage: "))
degree_t = input("Degree Type (Sci&Tech/Comm&Mgmt/Others): ")
workex = input("Work Experience (Yes/No): ")
etest_p = float(input("E-test Percentage: "))
specialisation = input("Specialisation (Mkt&HR/Mkt&Fin): ")
mba_p = float(input("MBA Percentage: "))

# Create input dataframe
new_data = pd.DataFrame({
    'ssc_p': [ssc_p],
    'hsc_p': [hsc_p],
    'degree_p': [degree_p],
    'etest_p': [etest_p],
    'mba_p': [mba_p],
    'gender_M': [1 if gender == 'M' else 0],
    'ssc_b_Others': [1 if ssc_b == 'Others' else 0],
    'hsc_b_Others': [1 if hsc_b == 'Others' else 0],
    'hsc_s_Commerce': [1 if hsc_s == 'Commerce' else 0],
    'hsc_s_Science': [1 if hsc_s == 'Science' else 0],
    'degree_t_Comm&Mgmt': [1 if degree_t == 'Comm&Mgmt' else 0],
    'degree_t_Sci&Tech': [1 if degree_t == 'Sci&Tech' else 0],
    'workex_Yes': [1 if workex == 'Yes' else 0],
    'specialisation_Mkt&HR': [1 if specialisation == 'Mkt&HR' else 0]
})

# Align columns with training data
new_data = new_data.reindex(columns=X.columns, fill_value=0)

prediction = model.predict(new_data)

print("\nPlacement Prediction:")
if prediction[0] == 1:
    print("Student is LIKELY to be PLACED")
else:
    print("Student is LIKELY to be NOT PLACED")


```

## Output:
<img width="1287" height="824" alt="image" src="https://github.com/user-attachments/assets/448b1f56-42a1-441c-8af0-f2078d5aaa43" />
<img width="1284" height="820" alt="image" src="https://github.com/user-attachments/assets/4e47a31f-a473-4c31-ba99-fcd55c750f0c" />





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
