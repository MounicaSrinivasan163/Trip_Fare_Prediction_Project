# 🚗 TripFare : Predicting Urban Taxi Fare with Machine Learning

**Streamlit app** : https://tripfarepredictionproject-c9vhppujsac8cayt3m6oan.streamlit.app/

## Description
As a Data Analyst at an urban mobility analytics firm, this project aims to unlock insights from real-world taxi trip data to enhance fare estimation systems and promote pricing transparency for passengers. Using historical taxi trip records from a metropolitan transportation network, the goal is to build a predictive model that accurately estimates the total taxi fare amount based on various ride-related features.

The project covers data preprocessing, feature engineering, exploratory data analysis (EDA), model training and evaluation of multiple regression models, hyperparameter tuning, and deployment of the best model using Streamlit.

---

## Domain
Urban Transportation Analytics & Predictive Modeling

---

## Skills Gained
- Exploratory Data Analysis (EDA)  
- Data Cleaning and Preprocessing  
- Data Visualization with Matplotlib & Seaborn  
- Feature Engineering  
- Regression Model Building  
- Model Evaluation & Comparison  
- Hyperparameter Tuning  
- Streamlit App Deployment  

---

## Problem Type
Supervised Machine Learning — Regression  
**Target Variable:** `total_amount`

---

## Dataset Columns

| Column Name           | Description                                      | Type         |
|-----------------------|------------------------------------------------|--------------|
| VendorID              | ID of the taxi provider                          | Categorical  |
| tpep_pickup_datetime  | Date and time when the trip started              | Continuous   |
| tpep_dropoff_datetime | Date and time when the trip ended                | Continuous   |
| passenger_count       | Number of passengers in the taxi                  | Categorical  |
| pickup_longitude      | Longitude where the passenger was picked up      | Continuous   |
| pickup_latitude       | Latitude where the passenger was picked up       | Continuous   |
| RatecodeID            | Type of rate (e.g., standard, JFK, negotiated)   | Categorical  |
| store_and_fwd_flag    | Whether trip data was stored and forwarded        | Categorical  |
| dropoff_longitude     | Longitude where the passenger was dropped off    | Continuous   |
| dropoff_latitude      | Latitude where the passenger was dropped off     | Continuous   |
| payment_type          | Payment method used                               | Categorical  |
| fare_amount           | Base fare amount charged                          | Continuous   |
| extra                 | Extra charges (e.g., peak time, night surcharge) | Continuous   |
| mta_tax               | MTA (Metropolitan Transportation Authority) tax | Continuous   |
| tip_amount            | Tip amount paid by the passenger                  | Continuous   |
| tolls_amount          | Toll charges (bridge/tunnel tolls)                 | Continuous   |
| improvement_surcharge | Flat fee surcharge (usually $0.30)                | Continuous   |
| total_amount          | Total trip amount including all fees and tips (Target) | Continuous |

---

## Tools Used
- Python 3.x  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost  
- Streamlit  

---

## Potential Use Cases
1. **Ride-Hailing Services** – Provide fare estimates before booking rides.  
2. **Driver Incentive Systems** – Suggest optimal locations and times for higher earnings.  
3. **Urban Mobility Analytics** – Analyze fare trends by time, location, and trip type.  
4. **Travel Budget Planners** – Predict estimated trip fares for tourists.  
5. **Taxi Sharing Apps** – Enable dynamic pricing for shared rides.  

---

## Data Pipeline

1. **Data Collection**  
   - Download dataset and load using Pandas.

2. **Data Understanding**  
   - Explore dataset shape, data types, duplicates, and missing values.

3. **Feature Engineering**  
   - Derive new features like `trip_distance` (using Haversine formula), `pickup_day` (weekday/weekend), `am/pm`, `is_night` (late-night trips), and convert timestamps to local time zones.

4. **Exploratory Data Analysis (EDA)**  
   - Analyze fare vs distance, fare vs passenger count, detect and handle outliers, study distributions over time, and visualize peak demand periods.

5. **Data Transformation**  
   - Handle outliers (Z-score/IQR), fix skewness in continuous variables, encode categorical features.

6. **Feature Selection**  
   - Use correlation analysis, Chi-Square tests, and feature importance from models to select relevant features.

7. **Model Building**  
   - Train multiple regression models (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting).  
   - Compare models using R², MSE, RMSE, MAE metrics.

8. **Hyperparameter Tuning**  
   - Optimize best model using GridSearchCV or RandomizedSearchCV.

9. **Finalize Best Model**  
   - Select the best model based on evaluation metrics and save it (pickle format).

10. **Deployment**  
    - Build a Streamlit UI for users to input trip details and get fare predictions.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/tripfare.git
cd tripfare

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
