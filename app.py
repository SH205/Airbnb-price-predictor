# Imports
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

model = joblib.load("airbnb_price_predictor.pkl")
model_features = joblib.load("model_features.pkl")

# -------------------- Page Configuration --------------------

st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -------------------- Custom CSS --------------------

st.markdown("""
<style>
.main {
    background-color: #fafafa;
}

.metric-card{
    background:#fff;
    padding:20px;
    border-radius:15px;
    box-shadow:0 2px 10px rgba(0,0,0,.10);
    margin-bottom:15px;
}

.section-header{
    font-size:28px;
    font-weight:bold;
    color:#FF5A5F;
    margin-top:15px;
}

.sub-header{
    font-size:18px;
    color:#555;
}

.prediction-box{
    background:#e8f5e9;
    padding:25px;
    border-radius:15px;
    text-align:center;
    border:2px solid #4CAF50;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:50px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------

@st.cache_resource
def load_model():

    return joblib.load( "airbnb_price_predictor.pkl")

# -------------------- Load Model Features --------------------

@st.cache_resource
def load_features():

    return joblib.load("model_features.pkl")

# -------------------- Load Dataset --------------------

@st.cache_data
def load_dataset():
    return pd.read_csv("cleaned_dataset.csv")

# Load everything
model = load_model()
model_features = load_features()
df = load_dataset()

# -------------------- Helper Functions --------------------

def calculate_stay_length(checkin_date, checkout_date):
    return (checkout_date - checkin_date).days

def get_checkin_month(checkin_date):
    return checkin_date.month

def predict_price(model, input_df):
    prediction_log = model.predict(input_df)
    return np.expm1(prediction_log)[0]

# -------------------- Sidebar --------------------

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_Bélo.svg",
        width=180
    )

    st.title("Airbnb Price Predictor")
    st.markdown("---")
    st.write( "Predict Airbnb nightly prices using a Random Forest machine learning model.")
    st.markdown("---")

    prediction_mode = st.radio(
        "Prediction Type",
        ["Existing Listing", "New Listing"], horizontal=True
    )

    st.markdown("---")
    st.info("""
**Existing Listing**

Uses review scores and listing information.

---

**New Listing**

Automatically fills unavailable historical information.
""")

    st.markdown("---")
    st.caption("Created using Streamlit")

# -------------------- Main Page --------------------

st.title("🏠 Airbnb Price Predictor")

st.write("""
Estimate the nightly price of an Airbnb listing using a machine learning model
trained on approximately **21,500** New York City Airbnb listings.

Fill in the information below and click **Predict Price**.
""")

# -------------------- Tabs --------------------

tab1, tab2 = st.tabs(["Prediction", "About Model"])

# -------------------- About Model --------------------

with tab2:

    st.header("About This Model")

    st.markdown("""
This application uses a **Random Forest Regressor** trained on Airbnb listing data.

### Model Performance

| Metric | Value |
|--------|-------|
| **MAE** | **$86.92** |
| **RMSE** | **$347.89** |
| **R²** | **0.453** |

### Features Used

The model considers:

- Property characteristics
- Geographic location
- Room type
- Amenities
- Review scores
- Stay length
- Check-in month

The preprocessing pipeline is automatically applied before every prediction.
""")


# ==========================================================
# Prediction Tab
# ==========================================================

with tab1:


    # ======================================================
    # PROPERTY INFORMATION
    # ======================================================

    st.markdown(
        '<p class="section-header">🏡 Property Information</p>',
        unsafe_allow_html=True
    )


    with st.container(border=True):

        col1, col2, col3 = st.columns(3)


        with col1:

            accommodates = st.number_input(
                "Guests",
                min_value=1,
                max_value=20,
                value=2
            )

            bedrooms = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=19,
                value=1
            )


        with col2:

            bathrooms = st.number_input(
                "Bathrooms",
                min_value=0,
                max_value=15,
                value=1,
                step=1
            )

            beds = st.number_input(
                "Beds",
                min_value=1,
                max_value=40,
                value=1
            )


        with col3:

            room_type = st.selectbox(
                "Room Type",
                sorted(df["room_type"].dropna().unique())
            )



    # ======================================================
    # LOCATION
    # ======================================================

    st.markdown(
        '<p class="section-header">📍 Location</p>',
        unsafe_allow_html=True
    )


    with st.container(border=True):

        col1, col2 = st.columns(2)


        with col2:

            borough = st.selectbox(
                "Neighborhood Group",
                sorted(
                    df["neighbourhood_group_cleansed"]
                    .dropna()
                    .unique()
                )
            )


        with col1:

            neighborhoods = sorted(
                df.loc[
                    df["neighbourhood_group_cleansed"] == borough,
                    "neighbourhood_cleansed"
                ]
                .dropna()
                .unique()
            )


            neighbourhood = st.selectbox(
                "Neighborhood",
                neighborhoods
            )


        location_row = df[
            df["neighbourhood_cleansed"] == neighbourhood
        ].iloc[0]


        latitude = location_row["latitude"]
        longitude = location_row["longitude"]




    # ======================================================
    # BOOKING INFORMATION
    # ======================================================

    st.markdown(
        '<p class="section-header">📅 Stay Information</p>',
        unsafe_allow_html=True
    )


    with st.container(border=True):

        col1, col2 = st.columns(2)


        with col1:

            stay_length = st.number_input(
                "Length of Stay (nights)",
                min_value=1,
                max_value=365,
                value=3
            )


        with col2:

            checkin_month = st.selectbox(
                "Check-in Month",
                options=list(range(1,13)),
                format_func=lambda x:
                    date(2025,x,1).strftime("%B")
            )

    # ======================================================
    # REVIEWS
    # ======================================================

    if prediction_mode == "Existing Listing":

        st.markdown(
            '<p class="section-header">⭐ Reviews</p>',
            unsafe_allow_html=True
        )


        with st.container(border=True):

            col1, col2, col3 = st.columns(3)


            with col1:

                review_scores_rating = st.number_input(
                    "Overall Rating",
                    min_value=1.0,
                    max_value=5.0,
                    value=4.8,
                    step=0.01
                )


            with col2:

                review_scores_cleanliness = st.number_input(
                    "Cleanliness",
                    min_value=1.0,
                    max_value=5.0,
                    value=4.8,
                    step=0.01
                )


            with col3:

                review_scores_location = st.number_input(
                    "Location Score",
                    min_value=1.0,
                    max_value=5.0,
                    value=4.8,
                    step=0.01
                )


    else:

        # Automatically assign average review scores
        review_scores_rating = df[
            "review_scores_rating"
        ].mean()


        review_scores_cleanliness = df[
            "review_scores_cleanliness"
        ].mean()


        review_scores_location = df[
            "review_scores_location"
        ].mean()
    


    # ======================================================
    # AMENITIES
    # ======================================================

    st.markdown(
        '<p class="section-header">✨ Amenities</p>',
        unsafe_allow_html=True
    )


    with st.container(border=True):


        amenity_list = {

            "wifi": "📶 WiFi",
            "kitchen": "🍳 Kitchen",
            "tv": "📺 TV",
            "air_conditioning": "❄ Air Conditioning",
            "washer": "🧺 Washer",
            "dryer": "🌬 Dryer",
            "dishwasher": "🍽 Dishwasher",
            "microwave": "🔥 Microwave",
            "refrigerator": "🧊 Refrigerator",
            "workspace": "💻 Workspace",
            "gym": "🏋 Gym",
            "pets_allowed": "🐶 Pets Allowed",
            "parking": "🅿 Parking",
            "elevator": "🛗 Elevator",
            "self_check_in": "🔑 Self Check-in"

        }


        amenities = {}


        cols = st.columns(5)


        for i, (key, label) in enumerate(amenity_list.items()):

            with cols[i % 5]:

                amenities[key] = st.checkbox(
                    label
                )


        amenities_count = sum(
            amenities.values()
        )


        # Selected amenities list

        selected_amenities = [

            label

            for key, label in amenity_list.items()

            if amenities[key]

        ]


        st.markdown("---")


        count_col, list_col = st.columns([1, 3])


        with count_col:

            st.metric(
                "Amenities Total",
                amenities_count
            )


        with list_col:

            if selected_amenities:

                st.markdown(
                    f"""
                    <div style="
                        background-color: rgba(115, 156, 121, 0.5); /* 50% opacity */
                        padding:12px;
                        border-radius:10px;
                        border:1px solid #c8e6c9;
                    ">
                    <b>Included Amenities:</b><br>
                    {"  ,  ".join(selected_amenities)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div style="
                        padding:12px;
                        border-radius:10px;
                    ">
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # ==========================================================
    # PREDICT PRICE
    # ==========================================================


    predict_container = st.container(border=True)


    with predict_container:


        predict_button = st.button(
            "🔮 Predict Airbnb Price",
            use_container_width=True
        )


        if predict_button:


            # ----------------------------------------------
            # Create input dataframe
            # ----------------------------------------------

            input_data = {

                "neighbourhood_cleansed": neighbourhood,
                "neighbourhood_group_cleansed": borough,

                "latitude": latitude,
                "longitude": longitude,

                "room_type": room_type,

                "accommodates": accommodates,
                "bathrooms": bathrooms,
                "bedrooms": bedrooms,
                "beds": beds,


                "review_scores_rating": review_scores_rating,
                "review_scores_cleanliness": review_scores_cleanliness,
                "review_scores_location": review_scores_location,


                "amenities_count": amenities_count,


                "wifi": int(amenities["wifi"]),
                "kitchen": int(amenities["kitchen"]),
                "tv": int(amenities["tv"]),
                "air_conditioning": int(amenities["air_conditioning"]),
                "washer": int(amenities["washer"]),
                "dryer": int(amenities["dryer"]),
                "dishwasher": int(amenities["dishwasher"]),
                "microwave": int(amenities["microwave"]),
                "refrigerator": int(amenities["refrigerator"]),
                "workspace": int(amenities["workspace"]),
                "gym": int(amenities["gym"]),
                "pets_allowed": int(amenities["pets_allowed"]),
                "parking": int(amenities["parking"]),
                "elevator": int(amenities["elevator"]),
                "self_check_in": int(amenities["self_check_in"]),


                "stay_length": stay_length,
                "checkin_month": checkin_month

            }


            input_df = pd.DataFrame(
                [input_data]
            )


            # ----------------------------------------------
            # Apply same encoding as training
            # ----------------------------------------------

            input_encoded = pd.get_dummies(
                input_df,
                drop_first=True
            )


            # Match training columns
            input_encoded = input_encoded.reindex(
                columns=model_features,
                fill_value=0
            )


            # ----------------------------------------------
            # Predict
            # ----------------------------------------------

            prediction_log = model.predict(
                input_encoded
            )


            prediction_price = np.expm1(
                prediction_log[0]
            )



            # ----------------------------------------------
            # Display Result
            # ----------------------------------------------
            st.markdown(
                f"""
                <div style="padding:15px;">
                🏡 Based on the selected property details, location,
                reviews, and amenities, the predicted Airbnb price is:
                </div>
                """,
                unsafe_allow_html=True)


            price_col1, price_col2 = st.columns(2)


            with price_col1:

                st.metric(
                    "Estimated Nightly Price",
                    f"${prediction_price:,.2f}"
                )


            with price_col2:

                st.metric(
                    "Estimated Total Price",
                    f"${prediction_price * stay_length:,.2f}"
                )

            st.success(
                "Prediction Complete!"
            )
