# Flight Price & Customer Satisfaction Prediction
A comprehensive end-to-end data science solution combining regression and classification use cases, deployed using Streamlit with integrated MLflow tracking.

## Project 1: Flight Price Prediction (Regression)
Domain: Travel and Tourism
Goal: Predict flight ticket prices based on various travel details like route, airline, and timings.

Problem Statement
Build a regression model to predict flight ticket prices using features such as departure time, airline, route, etc. Create a Streamlit app where users can input their travel details and receive a price prediction.

🧰 Skills & Tools
Python, Pandas, NumPy

Scikit-learn, XGBoost, Random Forest

MLflow (model tracking)

Streamlit (deployment)

Data Analysis & Visualization

🧭 Business Use Cases
Assist travelers in budgeting and planning trips.

Help travel agencies optimize pricing strategies.

Support airlines in trend analysis and dynamic pricing.

🛠️ Approach
Data Preprocessing

Clean and transform flight data.

Convert time columns and perform feature engineering.

Model Building

EDA and trend visualization.

Train multiple regression models (Linear, Random Forest, XGBoost).

Evaluate using RMSE and R².

MLflow Integration

Log metrics, parameters, and models.

Register best-performing models for production use.

Streamlit App

Visualize trends.

Allow user input for route, airline, time, and display price prediction.

📊 Dataset Features
Airline, Date_of_Journey, Source, Destination, Route

Dep_Time, Arrival_Time, Duration, Total_Stops

Additional_Info, Price

📁 Dataset: Flight_Price.csv
