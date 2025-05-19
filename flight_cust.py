#importing all necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
import plotly.express as px

# Load the registered model for fligth price prediction
model = mlflow.pyfunc.load_model(f"models:/Best_XGB_Model/2")

# Load cleaned dataset for consistency with model
@st.cache_data #result is cached. On future reruns (e.g., user clicks or page changes), 
#Streamlit skips reading the CSV again, and just uses the cached DataFrame — unless the file path or function changes.
def load_cleaned_data():
    return pd.read_csv("Cleaned_Flight_Price.csv")

cleaned_df = load_cleaned_data()
original_df=pd.read_csv("Flight_Price.csv")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home", 
    "Flight Price Trends Visualization", 
    "Flight Price Prediction", 
    "Customer Satisfaction Trends Visualization", 
    "Customer Satisfaction Prediction"
])
#----------------------------------------Homepage-----------------------------------------------------------------
if page == "Home":
    st.header("✈️ Flight Price & Customer Satisfaction Prediction App")
    st.markdown("""
    Welcome to the **Flight Price and Customer Satisfaction Prediction** app!

    🔍 Use the sidebar to navigate between:
    - **Flight Price Prediction**
    - **Customer Satisfaction Prediction**
    - **Visualizations** of trends and insights

    Start exploring to get flight fare estimates and see how airline services affect passenger satisfaction!
    """)
#----------------------------------Flight price visualization page----------------------------------------------------
elif page == "Flight Price Trends Visualization":
    st.header("📊 Flight Price Visualizations")

    st.markdown("##### Visualize price distributions, Compare average prices across Airlines, Source, and Destination")

    # Plot 1: Price Distribution Count (Histogram)
    fig1 = px.histogram(original_df, x="Price", nbins=50, title="Price Distribution Count")
    fig1.update_layout(height=400)
    st.plotly_chart(fig1, use_container_width=True)

    # Plot 2: Average Price by Airline (Bar Plot)
    airline_avg = original_df.groupby("Airline")["Price"].mean().sort_values(ascending=False).reset_index()
    fig2 = px.bar(airline_avg, x="Airline", y="Price", title="Average Price by Airline")
    fig2.update_layout(xaxis_tickangle=45, height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # Plot 3: Average Price by Source (Bar Plot)
    source_avg = original_df.groupby("Source")["Price"].mean().sort_values(ascending=False).reset_index()
    fig3 = px.bar(source_avg, x="Source", y="Price", title="Average Price by Source")
    fig3.update_layout(xaxis_tickangle=45, height=400)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Plot 4: Average Price by Destination (Bar Plot)
    destination_avg = original_df.groupby("Destination")["Price"].mean().sort_values(ascending=False).reset_index()
    fig4 = px.bar(destination_avg, x="Destination", y="Price", title="Average Price by Destination")
    fig4.update_layout(xaxis_tickangle=45, height=400)
    st.plotly_chart(fig4, use_container_width=True)
    
#----------------------------------Flight price Prediction page----------------------------------------------------
elif page == "Flight Price Prediction":
    st.header("Flight Price Prediction")
    st.write("Select Flight Details")

    input_dict = {}

    # Categorical options including dropped categories
    airline_options = ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
                       'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet',
                       'Trujet', 'Vistara', 'Vistara Premium economy']
    source_options = ['Bangalore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai']
    destination_options = ['Cochin', 'Bangalore','Delhi', 'Hyderabad', 'Kolkata']
    additional_info_options = [ 'No info','Business class', 'Change airports', 'In-flight meal not included', 'No check-in baggage included']

    # Row 1: Source and Destination
    col1, col2 = st.columns(2)
    with col1:
        selected_source = st.selectbox("Source", source_options)
    with col2:
        selected_destination = st.selectbox("Destination", destination_options)

    # Row 2: Day, Month, Stops
    col3, col4, col5 = st.columns(3)
    with col3:
        selected_day = st.selectbox("Day", list(range(1, 32)))
    with col4:
        selected_month = st.selectbox("Month", list(range(1, 13)))
    with col5:
        selected_stops = st.selectbox("Total Stops", [0, 1, 2, 3])

    # Row 3: Time details
    col6, col7, col8, col9 = st.columns(4)
    with col6:
        dep_hour = st.selectbox("Departure Hour", list(range(0, 24)))
    with col7:
        dep_min = st.selectbox("Departure Minute", list(range(0, 60, 5)))
    with col8:
        arr_hour = st.selectbox("Arrival Hour", list(range(0, 24)))
    with col9:
        arr_min = st.selectbox("Arrival Minute", list(range(0, 60, 5)))

    # Row 4: Duration
    col10, col11 = st.columns(2)
    with col10:
        duration_hrs = st.selectbox("Duration Hours", list(range(0, 25)))
    with col11:
        duration_mins = st.selectbox("Duration Minutes", list(range(0, 60, 5)))

    # Row 5: Airline and Additional Info
    col12, col13 = st.columns(2)
    with col12:
        selected_airline = st.selectbox("Airline", airline_options)
    with col13:
        selected_additional = st.selectbox("Additional Info", additional_info_options)

    # Fill input_dict
    input_dict['Day'] = selected_day
    input_dict['Month'] = selected_month
    input_dict['Total_Stops'] = selected_stops
    input_dict['Dep_hour'] = dep_hour
    input_dict['Dep_min'] = dep_min
    input_dict['Arrival_hour'] = arr_hour
    input_dict['Arrival_min'] = arr_min
    input_dict['Duration_hours'] = duration_hrs
    input_dict['Duration_mins'] = duration_mins

    for col in cleaned_df.columns:
        if col.startswith(('Airline_', 'Source_', 'Destination_', 'Additional_Info_')):
            input_dict[col] = 0

    if selected_airline != 'Airline_Air India':
        colname = f"Airline_{selected_airline}"
        if colname in input_dict:
            input_dict[colname] = 1

    if selected_source != 'Bangalore':
        colname = f"Source_{selected_source}"
        if colname in input_dict:
            input_dict[colname] = 1

    if selected_destination != 'Bangalore':
        colname = f"Destination_{selected_destination}"
        if colname in input_dict:
            input_dict[colname] = 1

    if selected_additional != 'No info':
        colname = f"Additional_Info_{selected_additional}"
        if colname in input_dict:
            input_dict[colname] = 1

    if st.button("Predict Price"):
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[cleaned_df.drop('Price', axis=1).columns]  # Align column order
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Flight Price: ₹{int(prediction):,}")

#--------------------------------------Customer Satisfaction Visualization page--------------------------------------------
elif page == "Customer Satisfaction Trends Visualization":
    st.header("📈 Customer Satisfaction Trends Visualization")
        
    @st.cache_data
    def load_satisfaction_data():
        df = pd.read_csv("cleaned_passenger_satisfaction.csv")
        return df.drop(columns=["Unnamed: 0"], errors="ignore")

    sat_df = load_satisfaction_data()
      
     # Satisfaction visualization
    satisfaction_map = {1: "Satisfied", 0: "Neutral or Dissatisfied"}
    sat_df["Satisfaction Label"] = sat_df["satisfaction"].map(satisfaction_map)

    # Plot histogram with new labels
    fig = px.histogram(
        sat_df,
        x="Satisfaction Label",
        color="Satisfaction Label",
        title="Distribution of Customer Satisfaction",
        category_orders={"Satisfaction Label": ["Neutral or Dissatisfied", "Satisfied"]},
        labels={"Satisfaction Label": "Customer Satisfaction"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # feature importance visualization 
    rating_features = [
        "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
        "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment",
        "On-board service", "Leg room service", "Baggage handling", "Checkin service",
        "Inflight service", "Cleanliness"
    ]
    avg_ratings = sat_df[rating_features].mean().sort_values(ascending=False)
    fig = px.bar(avg_ratings, x=avg_ratings.values, y=avg_ratings.index,
                 orientation='h', labels={'x': 'Average Rating', 'y': 'Service Feature'},
                 title='Average Ratings Across Service Features')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    import mlflow.xgboost
    import plotly.express as px
    import pandas as pd

    # ➕ Add feature importance chart
    st.markdown("### 🔍 Feature Importance from Model")

    # Load native XGBoost model from MLflow Model Registry
    model_uri = "models:/XGBoost_Default_Model/1"  # or update version as needed
    xgb_model = mlflow.xgboost.load_model(model_uri)

    # Get feature importances
    importances = xgb_model.feature_importances_
    features = xgb_model.get_booster().feature_names  # Use this if you set them manually
    # If not, use your own saved list like: features = X_train.columns

    # Create DataFrame for plotting
    importances_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    # Plot
    fig = px.bar(importances_df, x="Importance", y="Feature", orientation='h',
                title="🔍 Feature Importance in Predicting Satisfaction",
                labels={"Importance": "Importance Score", "Feature": "Feature"})
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
#--------------------------------------Customer Satisfaction Prediction page--------------------------------------------
elif page == "Customer Satisfaction Prediction":
    st.header("✈️ Customer Satisfaction Prediction")

    # Load MLflow model
    sat_model = mlflow.pyfunc.load_model("models:/XGBoost_Default_Model/1")  
    
    # Load cleaned dataset for column reference
    @st.cache_data
    def load_satisfaction_data():
        df = pd.read_csv("cleaned_passenger_satisfaction.csv")
        return df.drop(columns=["Unnamed: 0"], errors="ignore")  # Safe drop

    sat_df = load_satisfaction_data()
    # Row 1: Basic Customer Info
    st.markdown("##### Customer/Travel Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 10, 80, 30)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])

    # Row 2: Travel Info
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        travel_type = st.selectbox("Type of Travel", ["Business Travel", "Personal Travel"])
    with col5:
        travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    with col6:
        flight_distance = st.slider("Flight Distance", 100, 5000, 1000)
    with col7:
        delay_minutes = st.slider("Departure Delay (min)", 0, 1200, 0)

    # Input dictionary base
    input_dict = {
        "Age": age,
        "Flight Distance": flight_distance,
        "Departure Delay in Minutes": delay_minutes,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Customer Type_disloyal Customer": 1 if customer_type == "Disloyal Customer" else 0,
        "Type of Travel_Personal Travel": 1 if travel_type == "Personal Travel" else 0,
        "Class_Eco": 1 if travel_class == "Eco" else 0,
        "Class_Eco Plus": 1 if travel_class == "Eco Plus" else 0,
    }

    # Mapping display names to actual column names
    display_name_map = {
        "Online Boarding": "Online boarding",
        "Inflight WiFi": "Inflight wifi service",
        "Flight Entertainment": "Inflight entertainment",
        "Check-in Service": "Checkin service",
        "Cleanliness": "Cleanliness",
        "Baggage Handling": "Baggage handling",
        "Seat Comfort": "Seat comfort",
        "Leg Room": "Leg room service",
        "Onboard Service": "On-board service",
        "Inflight Service": "Inflight service",
        "Gate Location": "Gate location",
        "Dep/Arr Time Convenience": "Departure/Arrival time convenient",
        "Online Booking": "Ease of Online booking",
        "Food & Drink": "Food and drink"
    }

    # Row 3: Inflight Experience
    st.markdown("##### Ratings")
    row3_fields = ["Online Boarding", "Inflight WiFi", "Flight Entertainment", "Check-in Service", "Cleanliness"]
    cols3 = st.columns(5)
    for i, label in enumerate(row3_fields):
        with cols3[i]:
            input_dict[display_name_map[label]] = st.selectbox(label, list(range(6)), key=label)

    # Row 4: Seating & Comfort
    row4_fields = ["Baggage Handling", "Seat Comfort", "Leg Room", "Onboard Service", "Inflight Service"]
    cols4 = st.columns(5)
    for i, label in enumerate(row4_fields):
        with cols4[i]:
            input_dict[display_name_map[label]] = st.selectbox(label, list(range(6)), key=label)

    # Row 5: Ground & Booking Services
    row5_fields = ["Gate Location", "Dep/Arr Time Convenience", "Online Booking", "Food & Drink"]
    cols5 = st.columns(4)
    for i, label in enumerate(row5_fields):
        with cols5[i]:
            input_dict[display_name_map[label]] = st.selectbox(label, list(range(6)), key=label)

    # Prediction button
    if st.button("Predict Satisfaction"):
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[sat_df.drop("satisfaction", axis=1).columns]
        prediction = sat_model.predict(input_df)[0]
        result = "Satisfied 😊" if prediction == 1 else "Not Satisfied 😞"
        st.success(f"Predicted Customer Satisfaction: **{result}**")
        
       
