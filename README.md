<img width="1225" height="418" alt="image" src="https://github.com/user-attachments/assets/e6d56d5c-280b-4882-bd8d-dd56d2a78172" /># Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas

2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: S LOGESHWAR
RegisterNumber:  212225040204
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

# 1) Load dataset
csv_path = "Salary.csv"    # <-- Change path if needed
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found at: {csv_path}. Update the path.")

print("Dataset Loaded Successfully!\n")

# 2) Data exploration
print("Shape:", data.shape)
display(data.head())
print("\nInfo:")
display(data.info())
print("\nMissing Values:\n", data.isnull().sum())

# 3) Encode categorical column 'Position'
if "Position" in data.columns:
    le = LabelEncoder()
    data["Position"] = le.fit_transform(data["Position"])
    print("\nLabel Encoding Mapping (Position):")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(mapping)

# 4) Select features and target
X = data[["Position", "Level"]]
y = data["Salary"]

print("\nFeature Sample:")
display(X.head())
print("\nTarget Sample:")
display(y.head())

# 5) Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
print(f"\nTrain Size: {X_train.shape}, Test Size: {X_test.shape}")

# 6) Initialize and train Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=10)
dt.fit(X_train, Y_train)
print("\nModel Training Completed!")

# 7) Predict on test data
y_pred = dt.predict(X_test)
print("\nPredicted Salaries:", y_pred)

# 8) Evaluate model using R² Score
r2 = metrics.r2_score(Y_test, y_pred)
print(f"\nR² Score: {r2:.4f}")

# 9) Visualize Decision Tree
plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=["Position", "Level"], filled=True)
plt.title("Decision Tree Regressor for Salary Prediction")
plt.show()

# 10) Feature Importances
importances = pd.Series(dt.feature_importances_, index=["Position", "Level"])
print("\nFeature Importances:")
display(importances)



```

## Output:
<img width="1225" height="418" alt="image" src="https://github.com/user-attachments/assets/526e2385-b3c5-4f40-a268-8672d4749956" />
<img width="1218" height="415" alt="image" src="https://github.com/user-attachments/assets/e62de378-c08c-4fb9-b114-918f8b133eb3" />
<img width="1217" height="395" alt="image" src="https://github.com/user-attachments/assets/4835e23b-4019-489c-a34b-6a7f5ef41444" />
<img width="1223" height="228" alt="image" src="https://github.com/user-attachments/assets/75056396-6c4e-482a-aa7d-fd4d302c643f" />
<img width="1353" height="403" alt="image" src="https://github.com/user-attachments/assets/c23481fe-2d54-4c3c-8b3d-00da5bc10aad" />
<img width="1348" height="382" alt="image" src="https://github.com/user-attachments/assets/09087684-95c6-455d-b8d7-9194b9b04a64" />
<img width="1360" height="140" alt="image" src="https://github.com/user-attachments/assets/99dc19c8-bc80-4688-b451-f90edadda198" />
<img width="1360" height="140" alt="image" src="https://github.com/user-attachments/assets/6a17d117-cadc-4559-aaf3-2f71326f0720" />





## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
