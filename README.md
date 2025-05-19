# Flight Price & Customer Satisfaction Prediction
A comprehensive end-to-end data science solution combining regression and classification use cases, deployed using Streamlit with integrated MLflow tracking.

## Project 1: Flight Price Prediction (Regression)
### Domain: 
Travel and Tourism
### Goal: 
Predict flight ticket prices based on various travel details like route, airline, and timings.

### Problem Statement:
Build a regression model to predict flight ticket prices using features such as departure time, airline, route, etc. Create a Streamlit app where users can input their travel details and receive a price prediction.

### Skills & Tools
Python, Pandas, NumPy
Scikit-learn, XGBoost, Random Forest
MLflow (model tracking)
Streamlit (deployment)
Data Analysis & Visualization

### Business Use Cases
Assist travelers in budgeting and planning trips.
Help travel agencies optimize pricing strategies.
Support airlines in trend analysis and dynamic pricing.

### Approach
Data Preprocessing
Clean and transform flight data.
Convert time columns and perform feature engineering.
Model Building
EDA and trend visualization.
Train multiple regression models (Linear,decision tree, Random Forest,Gradientboost, XGBoost).
Evaluate using RMSE and R².
MLflow Integration
Log metrics, parameters, and models.
Register best-performing models for production use.
Streamlit App
Visualize trends.
Allow user input for route, airline, time, and display price prediction.

### Outcome:
Achieved high-accuracy predictions with XGBoost regression.
Successfully deployed the model in a dynamic UI with MLflow integration for model tracking.



## Project 2: Customer Satisfaction Prediction (Classification)
### Domain: 
Customer Experience
### Goal: 
Predict passenger satisfaction based on feedback and service experience.

### Problem Statement:
Train a classification model that predicts whether a passenger is satisfied based on inflight service ratings and travel details. Build a Streamlit app for users to input passenger info and get satisfaction predictions.

### Skills & Tools:
Python, Pandas, Scikit-learn
Logistic Regression, Random Forest, Gradient Boosting,XGBoost
MLflow (model tracking)
Streamlit (deployment)
Data Cleaning, EDA, Feature Engineering

### Business Use Cases
Identify unhappy customers early.
Improve services and increase retention.
Tailor marketing campaigns to satisfied segments.

### Approach
Data Preprocessing
Handle missing values and encode categorical variables.
Standardize numeric features.
Model Building
Explore feature relationships.
Train classifiers and evaluate with accuracy, F1-score.
MLflow Integration
Log models, confusion matrices, and performance metrics.
Streamlit App
Display satisfaction trends.
Accept input features and predict satisfaction.

### Outcome:
 Achieved high classification accuracy with Random Forest and Gradient Boosting.
 Delivered an end-to-end predictive tool embedded in a user-friendly Streamlit interface, fully tracked with MLflow.
