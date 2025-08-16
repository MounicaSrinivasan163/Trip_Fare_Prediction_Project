import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import geopandas as gpd
from datetime import datetime
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import pickle
import base64

# Set wide layout and title
st.set_page_config(page_title="Trip_Fare Prediction", layout="wide")
st.title("🚗 Trip Fare Prediction 💵")

# Set background image with overlay
def set_bg_from_local(image_path):
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.4)), url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        color: black;
    }}
    h1, h2, h3, h4, h5, h6, .stMarkdown, .css-1v0mbdj p {{
        color: black !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

image_path = r"D:\\DataScience\\Guvi_projects\\TripFare _ML\\photo.jpg"
set_bg_from_local(image_path)


# -------------------------
# Load Model and Preprocessing
# -------------------------
@st.cache_data
def load_model_and_preprocessing():
    with open('best_model_lgm.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessing.pkl', 'rb') as f:
        prep = pickle.load(f)
    return model, prep

model, preprocessing = load_model_and_preprocessing()
X_scaler = preprocessing['X_scaler']
y_scaler = preprocessing['y_scaler']
boxcox_lambdas = preprocessing['boxcox_lambdas']
boxcox_shifts = preprocessing['boxcox_shifts']

# -------------------------
# Inverse Box-Cox helper
# -------------------------
def inv_boxcox(y, lam):
    if lam == 0:
        return np.exp(y)
    else:
        return np.power(y * lam + 1, 1 / lam)

# -------------------------
# Prediction function
# -------------------------

def predict(input_df):
    global model, X_scaler, y_scaler
    
    # Step 1: scale input features
    X_scaled = X_scaler.transform(input_df)
    
    # Step 2: model prediction (scaled target space)
    y_pred_scaled = model.predict(X_scaled)
    
    # Step 3: inverse StandardScaler (back to raw values)
    y_pred_raw = y_scaler.inverse_transform(y_pred_scaled)
    
    # Step 4: format output
    target_cols = ['fare_amount', 'total_amount', 'duration', 'tip_amount']
    y_pred_final = {}
    for i, col in enumerate(target_cols):
        val = y_pred_raw[0, i]
        val = max(val, 0.0)  # no negatives
        
        # round appropriately
        if col == 'duration':
            y_pred_final[col] = round(val, 1)  # minutes
        else:
            y_pred_final[col] = round(val, 2)  # money
    
    # Step 5: compute other_charges
    other_charges = y_pred_final['total_amount'] - y_pred_final['fare_amount']
    y_pred_final['other_charges'] = round(max(other_charges, 0.0), 2)
    
    return y_pred_final



# Load GeoJSON and extract centroids and area names
geojson_path = 'NYC_NTA.geojson'

@st.cache_data
def load_area_centroids(geojson_path):
    gdf = gpd.read_file(geojson_path)
    area_column = 'ntaname'  # adjust if your geojson column is different
    gdf['centroid'] = gdf.geometry.centroid
    centroids_dict = {
        row[area_column]: (row['centroid'].y, row['centroid'].x)
        for idx, row in gdf.iterrows()
    }
    return centroids_dict

area_centroids = load_area_centroids(geojson_path)
area_list = sorted(area_centroids.keys())

st.title("ML Fare Prediction App")
tabs = st.tabs(["About", "Customer View", "Admin View"])

with tabs[0]:
    st.header("About this project")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ## Description
        As a Data Analyst at an urban mobility analytics firm, this project aims to unlock insights from real-world taxi trip data to enhance fare estimation systems and promote pricing transparency for passengers. Using historical taxi trip records from a metropolitan transportation network, the goal is to build a predictive model that accurately estimates the total taxi fare amount based on various ride-related features.

        The project covers data preprocessing, feature engineering, exploratory data analysis (EDA), model training and evaluation of multiple regression models, hyperparameter tuning, and deployment of the best model using Streamlit.
        """)
        st.markdown("### Domain\nUrban Transportation Analytics & Predictive Modeling")
        st.markdown("""
        ### Tools Used
        - Python 3.x  
        - Pandas, NumPy  
        - Matplotlib, Seaborn  
        - Scikit-learn  
        - XGBoost  
        - Streamlit
        """)
    with col2:
        st.markdown("""
        ### Uses of This Project
        - Estimate taxi fares before booking.
        - Help drivers find high-earning times/places.
        - Analyze fare trends for urban planning.
        - Assist travelers with trip budgeting.
        - Enable dynamic pricing for shared rides.
        - Detect fare anomalies or fraud.
        - Increase pricing transparency for customers.
        """)

with tabs[1]:
    st.header("Customer Input")
    st.markdown("Please enter customer details below:")

    # Initialize session_state defaults
    if 'predict_clicked' not in st.session_state:
        st.session_state.predict_clicked = False

    # Define defaults or get previous session values
    vendor_default = st.session_state.get('vendor_id', 1)
    passenger_default = st.session_state.get('passenger_count', 1)
    ratecode_default = st.session_state.get('ratecode_id', 1)
    store_flag_default = st.session_state.get('store_and_fwd_flag_str', 'yes')
    payment_default = st.session_state.get('payment_type', 1)
    pickup_area_default = st.session_state.get('pickup_area', area_list[0])
    dropoff_area_default = st.session_state.get('dropoff_area', area_list[0])
    pickup_time_default = st.session_state.get('pickup_time', datetime.now().time())
    pickup_date_default = st.session_state.get('pickup_date', datetime.now().date())

    col1, col2, col3 = st.columns(3)

    with col1:
        customer_name = st.text_input("Customer Name", value=st.session_state.get('customer_name', ''))
        vendor_id = st.selectbox("VendorID", options=[1, 2], index=[1, 2].index(vendor_default))
        passenger_count = st.selectbox("Passenger Count", options=[1, 2, 3, 4, 5, 6], index=[1, 2, 3, 4, 5, 6].index(passenger_default))
        ratecode_id = st.selectbox("RatecodeID", options=[1, 2, 3, 4, 5, 6, 99], index=[1, 2, 3, 4, 5, 6, 99].index(ratecode_default))

    with col2:
        store_and_fwd_flag_str = st.selectbox("Store and Forward Flag", options=["yes", "no"], index=["yes", "no"].index(store_flag_default))
        payment_type = st.selectbox("Payment Type", options=[1, 2, 3, 4], index=[1, 2, 3, 4].index(payment_default))
        pickup_area = st.selectbox("Pickup Area", options=area_list, index=area_list.index(pickup_area_default))

    with col3:
        dropoff_area = st.selectbox("Dropoff Area", options=area_list, index=area_list.index(dropoff_area_default))
        pickup_time = st.time_input("Pickup Time", value=pickup_time_default)
        pickup_date = st.date_input("Pickup Date", value=pickup_date_default)

    # Save inputs to session_state
    st.session_state.update({
        'customer_name': customer_name,
        'vendor_id': vendor_id,
        'passenger_count': passenger_count,
        'ratecode_id': ratecode_id,
        'store_and_fwd_flag_str': store_and_fwd_flag_str,
        'payment_type': payment_type,
        'pickup_area': pickup_area,
        'dropoff_area': dropoff_area,
        'pickup_time': pickup_time,
        'pickup_date': pickup_date
    })

    # Convert store_and_fwd_flag to int
    store_and_fwd_flag = 1 if store_and_fwd_flag_str == "yes" else 0

    # Map area names to numeric IDs
    area_to_id = {area: idx for idx, area in enumerate(area_list)}
    pickup_area_id = area_to_id.get(pickup_area, -1)
    dropoff_area_id = area_to_id.get(dropoff_area, -1)

    # Get lat/lon and calculate distance
    pickup_lat, pickup_long = area_centroids.get(pickup_area, (0.0, 0.0))
    drop_lat, drop_long = area_centroids.get(dropoff_area, (0.0, 0.0))
    distance_km = geodesic((pickup_lat, pickup_long), (drop_lat, drop_long)).km

    # Extract time features
    hour = pickup_time.hour
    is_peak_time = 1 if 5 <= hour <= 22 else 0
    day_of_week = pickup_date.weekday()
    day = pickup_date.day
    month = pickup_date.month

    # Layout: map on left, results on right
    map_col, result_col = st.columns([1, 1])

    # Build Folium map
    mid_lat, mid_long = (pickup_lat + drop_lat) / 2, (pickup_long + drop_long) / 2
    m = folium.Map(location=[mid_lat, mid_long], zoom_start=12)
    folium.Marker([pickup_lat, pickup_long], tooltip="Pickup", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker([drop_lat, drop_long], tooltip="Dropoff", icon=folium.Icon(color='red')).add_to(m)
    folium.PolyLine(locations=[[pickup_lat, pickup_long], [drop_lat, drop_long]], color='blue', weight=5).add_to(m)

    # Display map and distance
    with map_col:
        st.markdown(f"### Calculated Trip Distance: **{distance_km:.2f} km**")
        st_folium(m, width=500, height=400)

    # Display prediction KPIs after button press
    with result_col:
        if st.button("Predict Fare Details"):
            st.session_state.predict_clicked = True

            input_dict = {
                'VendorID': [vendor_id],
                'passenger_count': [passenger_count],
                'RatecodeID': [ratecode_id],
                'store_and_fwd_flag': [store_and_fwd_flag],
                'payment_type': [payment_type],
                'extra': [0],
                'mta_tax': [0],
                'tolls_amount': [0],
                'improvement_surcharge': [0],
                'pickup_area': [pickup_area_id],
                'dropoff_area': [dropoff_area_id],
                'Day': [day],
                'Month': [month],
                'day_of_week': [day_of_week],
                'hour': [hour],
                'trip_distance_km': [distance_km],
                'is_peak_time': [is_peak_time]
            }

            input_df = pd.DataFrame(input_dict)

            try:
                predictions = predict(
                    input_df)
                  

                st.success("Predicted Fare Details:")
                # Metrics in columns
                kpi_cols = st.columns(3)
                kpi_cols[0].metric("Fare Amount", f"${predictions['fare_amount']:.2f}")
                kpi_cols[1].metric("Other Charges", f"${predictions['other_charges']:.2f}")
                kpi_cols[2].metric("Total Amount", f"${predictions['total_amount']:.2f}")

                kpi_cols2 = st.columns(3)
                kpi_cols2[0].metric("Duration (mins)", f"{predictions['duration']:.1f}")
                kpi_cols2[1].metric("Tip Amount", f"${predictions['tip_amount']:.2f}")
                kpi_cols2[2].metric("Pickup Area", pickup_area)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        else:
            if st.session_state.get('predict_clicked', False):
                st.info("Please click 'Predict Fare Details' to see results.")


# Load the dataset for admin view
df = pd.read_csv("taxi_fare_with_areas.csv")

with tabs[2]:
    st.header("Admin View")
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Bar chart of revenue by region
        st.subheader("Top 10 Regions by Revenue")
        revenue_by_region = (
            df.groupby('pickup_area')['total_amount']
              .mean()
              .reset_index()
              .sort_values(by='total_amount', ascending=False)
              .head(10)
        )
        fig2 = px.bar(revenue_by_region, x='pickup_area', y='total_amount', title='Average Revenue by Region')
        st.plotly_chart(fig2, use_container_width=True)
        
        # 2. Map visualization of top 10 busiest routes
        st.subheader("Top 10 Busiest Routes Map")
        top_routes = (
            df.groupby(['pickup_area', 'dropoff_area'])
              .size()
              .reset_index(name='count')
              .sort_values(by='count', ascending=False)
              .head(10)
        )

        coords = (
            df.groupby(['pickup_area', 'dropoff_area'])
              .agg({
                  'pickup_latitude': 'mean',
                  'pickup_longitude': 'mean',
                  'dropoff_latitude': 'mean',
                  'dropoff_longitude': 'mean'
              })
              .reset_index()
        )

        top_routes = top_routes.merge(coords, on=['pickup_area', 'dropoff_area'])

        m = folium.Map(
            location=[40.7128, -74.0060],  # Centered on NYC
            tiles='OpenStreetMap',
            zoom_start=12
        )

        for _, row in top_routes.iterrows():    
            folium.Marker(
                location=[row['pickup_latitude'], row['pickup_longitude']],
                tooltip=f"Pickup: {row['pickup_area']}",
                icon=folium.Icon(color='green')
            ).add_to(m)
            folium.Marker(
                location=[row['dropoff_latitude'], row['dropoff_longitude']],
                tooltip=f"Dropoff: {row['dropoff_area']}",
                icon=folium.Icon(color='red')
            ).add_to(m)
            folium.PolyLine(
                locations=[
                    [row['pickup_latitude'], row['pickup_longitude']],
                    [row['dropoff_latitude'], row['dropoff_longitude']]
                ],
                color='blue',
                weight=2
            ).add_to(m)
        
        st_folium(m, width=700, height=500)
    
    with col2:
        # 3. Bar chart of other charges by region
        st.subheader("Top 10 Regions by Other Charges")
        df['other_charges'] = df['total_amount'] - df['fare_amount']
        other_charges_by_region = (
            df.groupby('pickup_area')['other_charges']
              .mean()
              .reset_index()
              .sort_values(by='other_charges', ascending=False)
              .head(10)
        )
        fig3 = px.bar(other_charges_by_region, x='pickup_area', y='other_charges', title='Average Other Charges by Region')
        st.plotly_chart(fig3, use_container_width=True)

        # 4. Pie chart showing distribution of payment types
        st.subheader("Payment Type Distribution")
        payment_counts = df['payment_type'].value_counts().reset_index()
        payment_counts.columns = ['payment_type', 'count']
        fig4 = px.pie(payment_counts, names='payment_type', values='count', title='Payment Type Distribution')
        st.plotly_chart(fig4, use_container_width=True)
