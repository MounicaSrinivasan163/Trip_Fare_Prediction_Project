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

image_path = r"D:\\DataScience\\Guvi_projects\\TripFare _ Predicting Urban Taxi Fare with Machine Learning\\photo.jpg"
set_bg_from_local(image_path)

# Load your trained model
@st.cache_data
def load_model():
    with open('xgb_multioutput_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

def predict(input_df):
    preds = model.predict(input_df)
    columns = ['other_charges', 'fare_per_km', 'fare_amount', 'total_amount', 'duration', 'tip_amount']
    pred_df = pd.DataFrame(preds, columns=columns)
    return pred_df

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

    # Initialize session_state defaults (only once)
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
        passenger_count = st.selectbox("Passenger Count", options=[1, 5, 3, 2, 4, 6], index=[1,5,3,2,4,6].index(passenger_default))
        ratecode_id = st.selectbox("RatecodeID", options=[1, 5, 2, 3, 4, 6, 99], index=[1,5,2,3,4,6,99].index(ratecode_default))

    with col2:
        store_and_fwd_flag_str = st.selectbox("Store and Forward Flag", options=["yes", "no"], index=["yes","no"].index(store_flag_default))
        payment_type = st.selectbox("Payment Type", options=[1, 2, 3, 4], index=[1,2,3,4].index(payment_default))
        pickup_area = st.selectbox("Pickup Area", options=area_list, index=area_list.index(pickup_area_default))

    with col3:
        dropoff_area = st.selectbox("Dropoff Area", options=area_list, index=area_list.index(dropoff_area_default))
        pickup_time = st.time_input("Pickup Time", value=pickup_time_default)
        pickup_date = st.date_input("Pickup Date", value=pickup_date_default)

    # Save inputs to session_state
    st.session_state['customer_name'] = customer_name
    st.session_state['vendor_id'] = vendor_id
    st.session_state['passenger_count'] = passenger_count
    st.session_state['ratecode_id'] = ratecode_id
    st.session_state['store_and_fwd_flag_str'] = store_and_fwd_flag_str
    st.session_state['payment_type'] = payment_type
    st.session_state['pickup_area'] = pickup_area
    st.session_state['dropoff_area'] = dropoff_area
    st.session_state['pickup_time'] = pickup_time
    st.session_state['pickup_date'] = pickup_date

    # Convert store_and_fwd_flag to int
    store_and_fwd_flag = 1 if store_and_fwd_flag_str == "yes" else 0

    # Map area names to numeric IDs
    area_to_id = {area: idx for idx, area in enumerate(area_list)}
    pickup_area_id = area_to_id.get(pickup_area, -1)
    dropoff_area_id = area_to_id.get(dropoff_area, -1)

    # Get lat/lon and calculate distance in km
    pickup_lat, pickup_long = area_centroids.get(pickup_area, (0.0, 0.0))
    drop_lat, drop_long = area_centroids.get(dropoff_area, (0.0, 0.0))
    distance_km = geodesic((pickup_lat, pickup_long), (drop_lat, drop_long)).km

    # Extract time features
    hour = pickup_time.hour
    is_peak_time = 1 if 5 <= hour <= 22 else 0
    day_of_week = pickup_date.weekday()
    day = pickup_date.day
    month = pickup_date.month

    # Create two columns: one for map, one for results
    map_col, result_col1 = st.columns([1, 1])  # map wider than results

    # Build Folium map
    mid_lat = (pickup_lat + drop_lat) / 2
    mid_long = (pickup_long + drop_long) / 2
    m = folium.Map(location=[mid_lat, mid_long], zoom_start=12)

    folium.Marker(location=[pickup_lat, pickup_long], tooltip="Pickup", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(location=[drop_lat, drop_long], tooltip="Dropoff", icon=folium.Icon(color='red')).add_to(m)
    folium.PolyLine(locations=[[pickup_lat, pickup_long], [drop_lat, drop_long]], color='blue', weight=5).add_to(m)

    # Display map on left
    with map_col:
        st.markdown(f"### Calculated Trip Distance: **{distance_km:.2f} km**")
        st_folium(m, width=500, height=400)

    # Display prediction KPIs on right after button press
    with result_col1:
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
                predictions = predict(input_df)
                pred = predictions.iloc[0]

                st.success("Predicted Fare Details:")

                # Place this once at the top of your script or in your app initialization
                
                with st.container():

                    # Open div with CSS class
                    

                        # Metrics inside columns
                    col1, col2 = st.columns(2)
                    with col1:
                            st.metric("Other Charges", f"${pred['other_charges']:.2f}")
                    with col2:
                            st.metric("Fare per Km", f"${pred['fare_per_km']:.2f}")

                    col3, col4 = st.columns(2)
                    with col3:
                            st.metric("Fare Amount", f"${pred['fare_amount']:.2f}")
                    with col4:
                            st.metric("Total Amount", f"${pred['total_amount']:.2f}")

                    col5, col6 = st.columns(2)
                    with col5:
                            st.metric("Duration (mins)", f"{pred['duration']:.1f}")
                    with col6:
                            st.metric("Tip Amount", f"${pred['tip_amount']:.2f}")

                        # Close div
                    st.markdown('</div>', unsafe_allow_html=True)



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
