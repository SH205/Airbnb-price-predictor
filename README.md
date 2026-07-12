
# 🏙️ Airbnb Price Predictor — New York City

## Machine Learning Web Application

🔗 **Live Demo:** https://airbnb-price-predictor-vquefao5xqpsymklchjuid.streamlit.app/


---

📌 Project Overview

This project develops a machine learning application that predicts Airbnb nightly prices for listings in New York City.

The model uses property characteristics, location information, amenities, review scores, and booking details to estimate the expected nightly price of an Airbnb listing.

The goal is to understand the factors influencing NYC Airbnb prices and create an interactive prediction tool using Streamlit.

---

## 🎯 Project Goals

- Analyze factors that influence Airbnb pricing
- Build and compare regression models
- Develop an end-to-end machine learning pipeline
- Deploy a user-friendly price prediction application

---

## 📊 Dataset Features

The model uses Airbnb listing information including:

**Property Features**
- Guests accommodated
- Bedrooms
- Bathrooms
- Beds
- Room type

**Location Features**
- Neighborhood
- Latitude
- Longitude

**Review Features**
- Overall rating
- Cleanliness score
- Location score

**Amenities**
- WiFi
- Kitchen
- Air conditioning
- Washer/dryer
- Parking
- Workspace
- Gym
- Self check-in
- And more

**Booking Features**
- Length of stay
- Check-in month

---

## 🔎 Data Processing

Data preparation included:

- Removing unnecessary features
- Handling missing values
- Encoding categorical variables
- Creating amenity-based features
- Applying log transformation to price due to right-skewed distribution

Target transformation:

\[
log(1 + price)
\]

was used to improve model performance.

---

## 🤖 Machine Learning Models

Models evaluated:

- Linear Regression
- Ridge Regression
- LASSO Regression
- Random Forest Regressor
- HistGradientBoosting Regressor

Models were compared using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The final model was selected based on prediction performance and deployment efficiency.

---

## 🌐 Streamlit Application

The deployed application allows users to enter:

- Property details
- Location
- Available amenities
- Review scores
- Stay information

and receive an estimated Airbnb nightly price. 

---

## 🛠 Technologies Used

**Programming**
- Python

**Data**
- Pandas
- NumPy
- Scikit-learn

**Machine Learning**
- Regression Models
- Pipeline preprocessing
- Feature engineering

**Deployment**
- Streamlit Community Cloud

**Model Storage**
- Joblib
