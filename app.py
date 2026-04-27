"""
============================================================
  Smart House Price Predictor
  Author: Tanvir Mazharul
  GitHub: github.com/tanvirmazharul

  A Streamlit web app that predicts house prices using
  an ensemble ML model, with live charts, confidence
  intervals, and neighbourhood market comparison.

  Run with:  streamlit run app.py
============================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Smart House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-title {
      font-size: 2.2rem; font-weight: 700;
      color: #1a1a2e; margin-bottom: 0;
  }
  .sub-title {
      font-size: 1rem; color: #555; margin-bottom: 1.5rem;
  }
  .metric-card {
      background: #f0f4ff; border-radius: 12px;
      padding: 1rem 1.2rem; margin-bottom: 0.5rem;
  }
  .price-display {
      font-size: 3rem; font-weight: 800;
      color: #1D9E75; text-align: center;
  }
  .confidence {
      font-size: 1rem; color: #555; text-align: center;
  }
  .section-header {
      font-size: 1.1rem; font-weight: 600;
      color: #1a1a2e; margin-top: 1rem; margin-bottom: 0.3rem;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA GENERATION  (realistic European housing)
# ─────────────────────────────────────────────

@st.cache_data
def generate_dataset(n=3000):
    np.random.seed(99)

    CITIES = ["Helsinki", "Espoo", "Tampere", "Turku", "Oulu",
              "Berlin", "Munich", "Amsterdam", "Stockholm", "Copenhagen"]

    CITY_BASE = {
        "Helsinki": 4500, "Espoo": 4200, "Tampere": 2800, "Turku": 2600,
        "Oulu": 2200, "Berlin": 5500, "Munich": 8500, "Amsterdam": 7800,
        "Stockholm": 6200, "Copenhagen": 6800,
    }

    CONDITION = ["Poor", "Fair", "Good", "Excellent"]
    CONDITION_MULT = {"Poor": 0.75, "Fair": 0.90, "Good": 1.00, "Excellent": 1.18}

    PROPERTY_TYPE = ["Apartment", "Townhouse", "Detached House", "Studio"]
    TYPE_MULT = {"Studio": 0.70, "Apartment": 1.00, "Townhouse": 1.15, "Detached House": 1.35}

    cities     = np.random.choice(CITIES, n)
    prop_type  = np.random.choice(PROPERTY_TYPE, n, p=[0.45, 0.20, 0.25, 0.10])
    condition  = np.random.choice(CONDITION, n, p=[0.10, 0.25, 0.45, 0.20])
    size_sqm   = np.random.normal(75, 30, n).clip(20, 300).round(1)
    bedrooms   = np.random.choice([1, 2, 3, 4, 5], n, p=[0.20, 0.35, 0.28, 0.12, 0.05])
    bathrooms  = np.random.choice([1, 2, 3], n, p=[0.55, 0.35, 0.10])
    year_built = np.random.randint(1920, 2024, n)
    age        = 2024 - year_built
    has_garage = np.random.choice([0, 1], n, p=[0.45, 0.55])
    has_garden = np.random.choice([0, 1], n, p=[0.40, 0.60])
    has_balcony= np.random.choice([0, 1], n, p=[0.35, 0.65])
    floor_num  = np.random.randint(0, 15, n)
    distance_center = np.random.exponential(8, n).clip(0.5, 40).round(1)
    energy_rating = np.random.choice(["A", "B", "C", "D", "E"], n,
                                      p=[0.10, 0.20, 0.35, 0.25, 0.10])
    energy_mult = {"A": 1.08, "B": 1.04, "C": 1.00, "D": 0.96, "E": 0.91}

    prices = np.array([
        CITY_BASE[c] * size_sqm[i]
        * CONDITION_MULT[cond]
        * TYPE_MULT[ptype]
        * energy_mult[en]
        * (1 - age[i] * 0.0015)
        * (1 + garg * 0.04)
        * (1 + gard * 0.05)
        * (1 + balc * 0.025)
        * max(0.7, 1 - dist * 0.012)
        + np.random.normal(0, 15000)
        for i, (c, cond, ptype, en, garg, gard, balc, dist) in enumerate(
            zip(cities, condition, prop_type, energy_rating,
                has_garage, has_garden, has_balcony, distance_center))
    ]).clip(40000, 2500000)

    df = pd.DataFrame({
        "city": cities,
        "property_type": prop_type,
        "condition": condition,
        "size_sqm": size_sqm,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "age": age,
        "has_garage": has_garage,
        "has_garden": has_garden,
        "has_balcony": has_balcony,
        "floor_number": floor_num,
        "distance_to_center_km": distance_center,
        "energy_rating": energy_rating,
        "price_eur": prices.round(-3),
    })
    return df, CITY_BASE


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────

@st.cache_data
def prepare_features(df):
    df = df.copy()

    le_city     = LabelEncoder()
    le_type     = LabelEncoder()
    le_cond     = LabelEncoder()
    le_energy   = LabelEncoder()

    df["city_enc"]     = le_city.fit_transform(df["city"])
    df["type_enc"]     = le_type.fit_transform(df["property_type"])
    df["cond_enc"]     = le_cond.fit_transform(df["condition"])
    df["energy_enc"]   = le_energy.fit_transform(df["energy_rating"])

    df["price_per_sqm"]    = df["price_eur"] / df["size_sqm"]
    df["room_ratio"]       = df["bedrooms"] / df["size_sqm"].clip(1)
    df["age_squared"]      = df["age"] ** 2
    df["size_x_condition"] = df["size_sqm"] * df["cond_enc"]
    df["amenity_score"]    = df["has_garage"] + df["has_garden"] + df["has_balcony"]
    df["location_score"]   = 1 / (df["distance_to_center_km"] + 1)

    feature_cols = [
        "city_enc", "type_enc", "cond_enc", "energy_enc",
        "size_sqm", "bedrooms", "bathrooms", "age", "age_squared",
        "has_garage", "has_garden", "has_balcony", "floor_number",
        "distance_to_center_km", "room_ratio", "size_x_condition",
        "amenity_score", "location_score",
    ]
    return df, feature_cols, le_city, le_type, le_cond, le_energy


# ─────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────

@st.cache_resource
def train_model():
    df, city_base = generate_dataset(3000)
    df_feat, feature_cols, le_city, le_type, le_cond, le_energy = prepare_features(df)

    X = df_feat[feature_cols].values
    y = df_feat["price_eur"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gb  = GradientBoostingRegressor(n_estimators=300, learning_rate=0.07,
                                     max_depth=5, random_state=42)
    rf  = RandomForestRegressor(n_estimators=200, max_depth=12,
                                min_samples_leaf=4, random_state=42)

    gb.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Ensemble: weighted average
    gb_pred = gb.predict(X_test)
    rf_pred = rf.predict(X_test)
    ensemble_pred = 0.6 * gb_pred + 0.4 * rf_pred

    mae = mean_absolute_error(y_test, ensemble_pred)
    r2  = r2_score(y_test, ensemble_pred)

    return gb, rf, df, city_base, feature_cols, le_city, le_type, le_cond, le_energy, mae, r2


# ─────────────────────────────────────────────
#  PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict_price(gb, rf, le_city, le_type, le_cond, le_energy,
                  city, prop_type, condition, size_sqm, bedrooms,
                  bathrooms, year_built, has_garage, has_garden,
                  has_balcony, floor_num, distance, energy_rating):

    age        = 2024 - year_built
    city_enc   = le_city.transform([city])[0]
    type_enc   = le_type.transform([prop_type])[0]
    cond_enc   = le_cond.transform([condition])[0]
    energy_enc = le_energy.transform([energy_rating])[0]

    room_ratio       = bedrooms / max(size_sqm, 1)
    age_squared      = age ** 2
    size_x_condition = size_sqm * cond_enc
    amenity_score    = has_garage + has_garden + has_balcony
    location_score   = 1 / (distance + 1)

    X = np.array([[city_enc, type_enc, cond_enc, energy_enc,
                   size_sqm, bedrooms, bathrooms, age, age_squared,
                   has_garage, has_garden, has_balcony, floor_num,
                   distance, room_ratio, size_x_condition,
                   amenity_score, location_score]])

    gb_pred  = gb.predict(X)[0]
    rf_pred  = rf.predict(X)[0]
    ensemble = 0.6 * gb_pred + 0.4 * rf_pred

    low  = ensemble * 0.91
    high = ensemble * 1.09
    return ensemble, low, high, gb_pred, rf_pred


# ─────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────

def main():
    # Header
    st.markdown('<p class="main-title">🏠 Smart House Price Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">ML-powered price estimation for European housing market · by Tanvir Mazharul</p>', unsafe_allow_html=True)
    st.divider()

    # Load model
    with st.spinner("Loading ML models..."):
        gb, rf, df, city_base, feature_cols, le_city, le_type, le_cond, le_energy, mae, r2 = train_model()

    # Model stats banner
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model R² Score",    f"{r2:.3f}",    help="How well the model explains price variance")
    col2.metric("Mean Absolute Error", f"€{mae:,.0f}", help="Average prediction error")
    col3.metric("Training Records",  "3,000",        help="Houses used to train the model")
    col4.metric("Features Used",     "18",           help="Input variables the model considers")

    st.divider()

    # ── SIDEBAR: User Inputs
    st.sidebar.markdown("## 🏡 Property Details")
    st.sidebar.markdown("Fill in the details below to get a price estimate.")

    city = st.sidebar.selectbox("City", [
        "Helsinki", "Espoo", "Tampere", "Turku", "Oulu",
        "Berlin", "Munich", "Amsterdam", "Stockholm", "Copenhagen"
    ])

    prop_type = st.sidebar.selectbox("Property Type", [
        "Apartment", "Townhouse", "Detached House", "Studio"
    ])

    condition = st.sidebar.selectbox("Condition", ["Poor", "Fair", "Good", "Excellent"])

    energy_rating = st.sidebar.selectbox("Energy Rating", ["A", "B", "C", "D", "E"])

    st.sidebar.markdown("---")
    size_sqm  = st.sidebar.slider("Size (m²)",         20,  300, 75)
    bedrooms  = st.sidebar.slider("Bedrooms",           1,   5,   2)
    bathrooms = st.sidebar.slider("Bathrooms",          1,   3,   1)
    floor_num = st.sidebar.slider("Floor Number",       0,  15,   2)
    year_built = st.sidebar.slider("Year Built",     1920, 2023, 1990)
    distance  = st.sidebar.slider("Distance to City Centre (km)", 0.5, 40.0, 5.0, step=0.5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Amenities**")
    has_garage  = int(st.sidebar.checkbox("Garage",  value=True))
    has_garden  = int(st.sidebar.checkbox("Garden",  value=False))
    has_balcony = int(st.sidebar.checkbox("Balcony", value=True))

    predict_btn = st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True)

    # ── MAIN AREA
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        if predict_btn:
            price, low, high, gb_pred, rf_pred = predict_price(
                gb, rf, le_city, le_type, le_cond, le_energy,
                city, prop_type, condition, size_sqm, bedrooms,
                bathrooms, year_built, has_garage, has_garden,
                has_balcony, floor_num, distance, energy_rating
            )
            price_per_sqm = price / size_sqm

            # Price display
            st.markdown("### 💰 Estimated Price")
            st.markdown(f'<p class="price-display">€{price:,.0f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="confidence">Confidence range: €{low:,.0f} – €{high:,.0f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="confidence">Price per m²: €{price_per_sqm:,.0f}</p>', unsafe_allow_html=True)

            st.divider()

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=price / 1000,
                number={"prefix": "€", "suffix": "k", "font": {"size": 28}},
                delta={"reference": city_base[city] * size_sqm / 1000,
                       "relative": True, "valueformat": ".1%"},
                gauge={
                    "axis": {"range": [0, 2000], "tickprefix": "€", "ticksuffix": "k"},
                    "bar": {"color": "#1D9E75"},
                    "steps": [
                        {"range": [0,    400],  "color": "#e8f5e9"},
                        {"range": [400,  800],  "color": "#c8e6c9"},
                        {"range": [800,  1200], "color": "#a5d6a7"},
                        {"range": [1200, 2000], "color": "#81c784"},
                    ],
                    "threshold": {
                        "line": {"color": "#185FA5", "width": 3},
                        "thickness": 0.75,
                        "value": city_base[city] * size_sqm / 1000,
                    },
                },
                title={"text": f"Predicted vs City Average ({city})", "font": {"size": 14}},
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Model agreement
            st.markdown("**Model Agreement**")
            agree_df = pd.DataFrame({
                "Model": ["Gradient Boosting", "Random Forest", "Ensemble"],
                "Prediction (€)": [gb_pred, rf_pred, price],
            })
            fig_agree = px.bar(agree_df, x="Model", y="Prediction (€)",
                               color="Model",
                               color_discrete_map={
                                   "Gradient Boosting": "#185FA5",
                                   "Random Forest": "#1D9E75",
                                   "Ensemble": "#D85A30",
                               },
                               text_auto=".3s")
            fig_agree.update_layout(height=260, showlegend=False,
                                    margin=dict(t=10, b=10, l=10, r=10),
                                    yaxis_title="Price (EUR)")
            fig_agree.update_traces(textposition="outside")
            st.plotly_chart(fig_agree, use_container_width=True)

        else:
            st.info("👈 Fill in the property details in the sidebar and click **Predict Price** to get started.")
            st.markdown("### How it works")
            st.markdown("""
This app uses an **ensemble of two ML models** trained on 3,000 European property records:

- **Gradient Boosting** — captures complex non-linear patterns
- **Random Forest** — reduces variance through bagging
- **Ensemble** — combines both for the most accurate prediction

The models consider **18 features** including location, size, age, condition, amenities, energy rating, and distance to city centre.
            """)

    with right:
        st.markdown("### 📊 Market Insights")

        # City price comparison
        city_medians = df.groupby("city")["price_eur"].median().sort_values(ascending=True).reset_index()
        city_medians.columns = ["City", "Median Price (€)"]

        fig_cities = px.bar(city_medians, x="Median Price (€)", y="City",
                            orientation="h",
                            color="Median Price (€)",
                            color_continuous_scale="Blues",
                            text_auto=".3s")
        fig_cities.update_layout(
            title="Median Property Price by City",
            height=320,
            margin=dict(t=40, b=10, l=10, r=10),
            coloraxis_showscale=False,
        )
        fig_cities.update_traces(textposition="outside")
        st.plotly_chart(fig_cities, use_container_width=True)

        # Price vs Size scatter
        sample = df.sample(300, random_state=1)
        fig_scatter = px.scatter(
            sample, x="size_sqm", y="price_eur",
            color="property_type",
            hover_data=["city", "bedrooms", "condition"],
            labels={"size_sqm": "Size (m²)", "price_eur": "Price (€)",
                    "property_type": "Type"},
            color_discrete_sequence=["#185FA5", "#1D9E75", "#D85A30", "#BA7517"],
            opacity=0.65,
        )
        fig_scatter.update_layout(
            title="Price vs Size by Property Type",
            height=300,
            margin=dict(t=40, b=10, l=10, r=10),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Price distribution
        fig_dist = px.histogram(
            df, x="price_eur", nbins=50,
            labels={"price_eur": "Price (€)"},
            color_discrete_sequence=["#185FA5"],
            opacity=0.8,
        )
        fig_dist.update_layout(
            title="Overall Price Distribution",
            height=260,
            margin=dict(t=40, b=10, l=10, r=10),
            yaxis_title="Count",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align:center; color:#888; font-size:0.85rem;'>
    Built by <strong>Tanvir Mazharul</strong> · 
    Python · scikit-learn · Streamlit · Plotly &nbsp;|&nbsp;
    <a href='https://github.com/tanvirmazharul' target='_blank'>GitHub</a> &nbsp;|&nbsp;
    <a href='https://www.tanvirmazharul.com' target='_blank'>Portfolio</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
