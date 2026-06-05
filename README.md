Titanic Survival Prediction (Machine Learning Project)
Overview

This project builds a machine learning model to predict passenger survival on the Titanic using Logistic Regression. It includes data preprocessing, feature engineering, and model evaluation using scikit-learn.

The objective is to classify whether a passenger survived based on attributes such as age, ticket class, fare, and family relationships.

Dataset

The dataset used is titanic.csv, which contains passenger information including:

Age
Passenger Class (Pclass)
Fare
Number of siblings/spouses aboard
Number of parents/children aboard
Survival status (target variable)
Technologies Used
Python
Pandas
Scikit-learn

Key components from scikit-learn:

Logistic Regression
Pipeline
StandardScaler
train_test_split
accuracy_score
Data Preprocessing

The following preprocessing steps were applied:

Missing values in Age were filled using the median
Categorical transformation of Age into Age Groups:
Child
Teen
Young
Adult
Mapping Pclass into categorical labels:
rich
middle
poor
Feature creation:
Family_Size = Siblings/Spouses Aboard + Parents/Children Aboard + 1
Is_Alone = 1 if Family_Size is 1, otherwise 0
Fare was grouped into bins
One-hot encoding applied using get_dummies
Model Pipeline

A machine learning pipeline was used to ensure structured preprocessing and training:

Feature scaling using StandardScaler
Logistic Regression model with max iterations set to 1000
Training and Testing

The dataset was split into training and testing sets using an 80/20 ratio.

Training set: 80%
Testing set: 20%
Random state: 42
Evaluation Metric

The model is evaluated using accuracy score:

print("Accuracy", accuracy_score(y_test, y_pred))
How to Run the Project
1. Clone the repository
git clone https://github.com/your-username/titanic-ml-project.git
2. Install dependencies
pip install pandas scikit-learn
3. Run the script
python main.py
Results
Model: Logistic Regression
Task: Binary classification (Survived / Not Survived)
Metric: Accuracy score
Future Improvements
Experiment with advanced models such as Random Forest and XGBoost
Perform hyperparameter tuning
Add cross-validation
Analyze feature importance
Deploy as a web application using Flask or FastAPI
Key Learnings

This project demonstrates:

Practical feature engineering techniques
Importance of preprocessing pipelines
End-to-end machine learning workflow
Model evaluation fundamentals

Link to titanic.csv file (I do not own the file)
https://www.kaggle.com/datasets/yasserh/titanic-dataset
