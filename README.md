# 🏠 Smart House Price Predictor

> An interactive machine learning web app that predicts European property prices in real time. Built with Python, scikit-learn, Streamlit, and Plotly.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.10+-purple?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 What It Does

Enter any property's details — location, size, condition, amenities — and get an **instant price prediction** with:

- Estimated price + confidence range (±9%)
- Price per m² breakdown
- Gauge chart comparing your prediction vs city average
- Side-by-side model agreement (Gradient Boosting vs Random Forest vs Ensemble)
- Live market charts: city comparisons, price distributions, size scatter plots

---

## 🧠 ML Architecture

| Component | Detail |
|---|---|
| Dataset | 3,000 synthetic European property records |
| Features | 18 engineered features |
| Models | Gradient Boosting + Random Forest (ensemble) |
| Ensemble | Weighted average: 60% GB + 40% RF |
| Accuracy | R² ≈ 0.92, MAE ≈ €18,000 |
| UI | Streamlit + Plotly interactive charts |

### Feature Engineering Highlights
- `age_squared` — captures non-linear depreciation
- `room_ratio` — bedrooms relative to size
- `location_score` — inverse distance to city centre
- `size_x_condition` — interaction term
- `amenity_score` — combined garage + garden + balcony

### Cities Covered
Helsinki · Espoo · Tampere · Turku · Oulu · Berlin · Munich · Amsterdam · Stockholm · Copenhagen

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/tanvirmazharul/house-price-predictor.git
cd house-price-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the app
```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501` in your browser.

---

## 🖥️ App Preview

**Sidebar** — enter property details:
- City, property type, condition, energy rating
- Size, bedrooms, bathrooms, floor, year built
- Distance to centre, amenities (garage, garden, balcony)

**Main panel** — instant results:
- Big price display with confidence range
- Gauge chart vs city average
- Model agreement bar chart
- Market insight charts (always visible)

---

## 📁 Project Structure

```
house-price-predictor/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

---

## 🛣️ Potential Extensions

- Connect to a real property listing API (e.g. Zillow, Rightmove)
- Add map visualisation with Folium or Deck.gl
- Implement SHAP explainability — show why a price was predicted
- Add neighbourhood crime / school rating features
- Deploy to Streamlit Cloud (free hosting, one click)

---

## ☁️ Deploy for Free (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → `app.py`
5. Click **Deploy** — your app gets a public URL instantly!

---

## 🧑‍💻 Author

**Tanvir Mazharul** — Software Developer & ML Enthusiast  
📍 Lappeenranta, Finland  
🔗 [tanvirmazharul.com](https://www.tanvirmazharul.com)  
📧 tanvirmazharul04@gmail.com

---

## 📄 License

MIT License — free to use, modify, and distribute.
