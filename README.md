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
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# 1. Load Dataset
df = pd.read_csv("Placement_Data.csv") 
print("Dataset Preview:")
print(df.head())

# 2. Data Preprocessing
# Drop irrelevant column
df.drop(columns=['sl_no', 'salary'], inplace=True)

# Encode categorical variables
label_cols = [
    'gender', 'ssc_b', 'hsc_b', 'hsc_s',
    'degree_t', 'workex', 'specialisation', 'status'
]

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])
-
# 3. Split Features and Target
X = df.drop('status', axis=1)   # input features
y = df['status']                # target variable

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Model Evaluation
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Specificity calculation
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print("\nConfusion Matrix:")
print(cm)

print("\nModel Performance:")
print(f"Accuracy     : {accuracy:.2f}")
print(f"Precision    : {precision:.2f}")
print(f"Recall       : {recall:.2f}")
print(f"Specificity  : {specificity:.2f}")


# 7. Predict Placement for New Student

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

# Encode user input manually (same mapping as training)
input_data = pd.DataFrame([[
    1 if gender == 'M' else 0,
    ssc_p,
    1 if ssc_b == 'Central' else 0,
    hsc_p,
    1 if hsc_b == 'Central' else 0,
    0 if hsc_s == 'Arts' else 1,
    degree_p,
    1 if degree_t == 'Sci&Tech' else 0,
    1 if workex == 'Yes' else 0,
    etest_p,
    1 if specialisation == 'Mkt&Fin' else 0,
    mba_p
]], columns=X.columns)

prediction = model.predict(input_data)

print("\nPlacement Prediction:")
if prediction[0] == 1:
    print("Student is LIKELY to be PLACED")
else:
    print("Student is LIKELY to be NOT PLACED")

```

## Output:
<img width="1000" height="583" alt="image" src="https://github.com/user-attachments/assets/1e93de53-0379-4879-8182-9a599b279719" />
<img width="996" height="580" alt="image" src="https://github.com/user-attachments/assets/33dce6f2-22a9-40b4-a7a6-d01cac2de904" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
