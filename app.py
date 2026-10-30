
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------
MODEL_PATH = Path(__file__).parent / "real_estate_price_model.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    model = None
    model_error = str(error)
else:
    model_error = None

# --------------------------------------------------
# 3. MAIN HEADING
# --------------------------------------------------
st.title("🏡 Real Estate Investment Advisor")
st.subheader("Property Price Estimation and Investment Analysis")

st.warning(
    "Experimental project: the current ML model has not "
    "demonstrated reliable property-price predictions. "
    "Do not use its results for actual investment decisions."
)

if model is None:
    st.error("The trained model could not be loaded.")
    if model_error:
        st.code(model_error)
    else:
        st.write(
            "Make sure real_estate_price_model.joblib is "
            "in the same folder as app.py."
        )
    st.stop()

st.success("Trained ML model loaded successfully.")
st.divider()

# --------------------------------------------------
# 4. PROPERTY DETAILS
# --------------------------------------------------
st.header("🏠 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox(
        "Select City",
        ["Bangalore", "Chennai", "Hyderabad", "Mumbai", "Delhi"]
    )

    property_type = st.selectbox(
        "Property Type",
        ["Apartment", "Independent House", "Villa"]
    )

    bhk = st.selectbox(
        "Number of Bedrooms (BHK)",
        [1, 2, 3, 4, 5]
    )

    size = st.number_input(
        "Property Size (sq. ft.)",
        min_value=100,
        max_value=10000,
        value=1200,
        step=100
    )

with col2:
    property_age = st.number_input(
        "Property Age (years)",
        min_value=0,
        max_value=100,
        value=5
    )

    furnished_status = st.selectbox(
        "Furnished Status",
        ["Unfurnished", "Semi-Furnished", "Furnished"]
    )

    floor_no = st.number_input(
        "Floor Number",
        min_value=0,
        max_value=100,
        value=2
    )

    total_floors = st.number_input(
        "Total Floors in Building",
        min_value=1,
        max_value=100,
        value=5
    )

st.subheader("Additional Property Features")

col3, col4 = st.columns(2)

with col3:
    nearby_schools = st.number_input(
        "Nearby Schools",
        min_value=0,
        max_value=50,
        value=3
    )

    nearby_hospitals = st.number_input(
        "Nearby Hospitals",
        min_value=0,
        max_value=50,
        value=2
    )

    transport = st.selectbox(
        "Public Transport Accessibility",
        ["Low", "Medium", "High"]
    )

    parking = st.selectbox(
        "Parking Space",
        ["No", "Yes"]
    )

with col4:
    security = st.selectbox(
        "Security",
        ["No", "Yes"]
    )

    amenities = st.selectbox(
        "Amenities",
        ["Basic", "Standard", "Premium"]
    )

    facing = st.selectbox(
        "Property Facing",
        ["North", "South", "East", "West"]
    )

    owner_type = st.selectbox(
        "Owner Type",
        ["Individual", "Builder", "Broker"]
    )

availability_status = st.selectbox(
    "Availability Status",
    ["Ready to Move", "Under Construction"]
)

annual_growth = st.slider(
    "Assumed Annual Price Growth (%)",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.5
)

st.caption(
    "The growth rate is an assumption for scenario analysis, "
    "not a market forecast."
)

st.divider()

# --------------------------------------------------
# 5. PROPERTY SUMMARY
# --------------------------------------------------
st.header("📋 Property Summary")

summary1, summary2, summary3 = st.columns(3)

summary1.metric("City", city)
summary2.metric("Property Type", property_type)
summary3.metric("Bedrooms", f"{bhk} BHK")

summary1.metric("Property Size", f"{size:,.0f} sq. ft.")
summary2.metric("Property Age", f"{property_age} years")
summary3.metric("Annual Growth Assumption", f"{annual_growth:.1f}%")

st.divider()

# --------------------------------------------------
# 6. PRICE PREDICTION
# --------------------------------------------------
st.header("📊 Price Estimation")

if st.button(
    "Predict Property Price",
    type="primary",
    key="predict_price"
):
    year_built = datetime.now().year - int(property_age)

    # Input columns must match the training pipeline.
    property_data = pd.DataFrame([{
        "City": city,
        "Property_Type": property_type,
        "BHK": int(bhk),
        "Size_in_SqFt": float(size),
        "Year_Built": year_built,
        "Furnished_Status": furnished_status,
        "Floor_No": int(floor_no),
        "Total_Floors": int(total_floors),
        "Nearby_Schools": int(nearby_schools),
        "Nearby_Hospitals": int(nearby_hospitals),
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Amenities": amenities,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability_status
    }])

    try:
        predicted_price = float(model.predict(property_data)[0])

        if not pd.notna(predicted_price):
            st.error("The model returned an invalid price estimate.")

        elif predicted_price < 0:
            st.error(
                "The model returned a negative price estimate. "
                "Please review the model before using predictions."
            )

        else:
            future_price = predicted_price * (
                1 + annual_growth / 100
            ) ** 5

            st.subheader("Estimated Results")

            result1, result2 = st.columns(2)

            result1.metric(
                "Model Price Estimate",
                f"₹{predicted_price:,.2f} Lakhs"
            )

            result2.metric(
                "Five-Year Scenario Value",
                f"₹{future_price:,.2f} Lakhs"
            )

            st.info(
                "The five-year scenario value is calculated by "
                "compounding your assumed annual growth rate for "
                "five years. It is not a market forecast."
            )

            st.warning(
                "Model evaluation: R² ≈ -0.0025 and MAE ≈ "
                "₹122.43 lakhs on the test set. The model has not "
                "shown reliable price predictions. Do not use "
                "these estimates for actual investment decisions."
            )

    except Exception as error:
        st.error("Prediction failed. Details are shown below.")
        st.code(str(error))
        
# --------------------------------------------------
# 6.1 MODEL PERFORMANCE AND LIMITATIONS
# --------------------------------------------------
with st.expander("📘 Understand the Model and Its Limitations"):

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Trained Model MAE", "₹122.43 Lakhs")

    with col2:
        st.metric("Mean Baseline MAE", "₹122.32 Lakhs")

    st.warning(
        "The trained model performs slightly worse than the "
        "mean baseline. Its predictions are not reliable."
    )

    st.subheader("Dataset Limitations")

    st.markdown("""
    - Property size has almost zero correlation with price.
    - Average prices are similar across BHK groups.
    - Average prices are similar across property types.
    - City-level average prices are also very similar.
    - The dataset may not represent realistic property-market patterns.
    """)

    st.subheader("Five-Year Scenario")

    st.info(
        "The five-year value is calculated using your assumed "
        "annual growth rate. It is a hypothetical scenario, "
        "not a market forecast."
    )

    st.caption(
        "Educational project only. Not intended for actual "
        "property valuation or investment recommendations."
    )


# --------------------------------------------------
# 7. FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Educational project | Machine Learning + Streamlit | "
    "Not financial advice"
)

# --------------------------------------------------
# 8. MODEL PERFORMANCE COMPARISON CHART
# --------------------------------------------------

st.header("📈 Model Performance Comparison")

# Actual evaluation results
evaluation_data = pd.DataFrame({
    "Model": ["Trained Model", "Mean Baseline"],
    "MAE": [122.4344, 122.3178],
    "RMSE": [141.3689, 141.1932],
    "R2 Score": [-0.002491, -0.0000013]
})

st.subheader("Mean Absolute Error (MAE)")
st.caption("Lower values indicate better performance.")

st.bar_chart(
    evaluation_data.set_index("Model")[["MAE"]]
)

st.subheader("Root Mean Squared Error (RMSE)")
st.caption("Lower values indicate better performance.")

st.bar_chart(
    evaluation_data.set_index("Model")[["RMSE"]]
)

st.subheader("R² Score")
st.caption("Higher values are better. A negative score indicates performance worse than the mean baseline.")

st.bar_chart(
    evaluation_data.set_index("Model")[["R2 Score"]]
)

st.warning(
    "The trained model performs slightly worse than the mean baseline. "
    "Further improvements to the dataset and features are needed."
)
