# crop_recommendation_app.py
import os
import pickle
import numpy as np
import streamlit as st
import requests

st.set_page_config(page_title="🌾 Smart Crop Recommendation", page_icon="🌱", layout="centered")

st.title("🌾 Smart Crop Recommendation System")
st.markdown("Get crop suggestions based on **State**, **Season**, and **Area**")


@st.cache_data(show_spinner=False)
def load_artifacts():
    required = ["model.pkl", "le_state.pkl", "le_season.pkl", "le_crop.pkl"]
    for f in required:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}\nPlease save the .pkl files in the app directory.")
    model = pickle.load(open("model.pkl", "rb"))
    le_state = pickle.load(open("le_state.pkl", "rb"))
    le_season = pickle.load(open("le_season.pkl", "rb"))
    le_crop = pickle.load(open("le_crop.pkl", "rb"))
    return model, le_state, le_season, le_crop

# ------------------------
# Weather helper (wttr.in)
# ------------------------
def get_weather_text(location):
    """Fetch simple weather summary using wttr.in (no API key required)."""
    try:
        loc = location.replace(" ", "+")
        url = f"https://wttr.in/{loc}?format=%C+%t+%m"  # condition + temp + moon
        r = requests.get(url, timeout=6)
        if r.status_code == 200 and r.text.strip():
            return r.text.strip()
        return "Weather data not available"
    except Exception:
        return "Weather data not available"

crop_details = {
    "Maize": {"fertilizer": "DAP, Urea, Potash", "pesticide": "Chlorpyrifos, Cypermethrin", "yield_per_acre": 25},
    "Soyabean": {"fertilizer": "SSP, Potash, Ammonium Sulphate", "pesticide": "Imidacloprid, Lambda-cyhalothrin", "yield_per_acre": 10},
    "Jowar": {"fertilizer": "Urea, DAP, Zinc Sulphate", "pesticide": "Dimethoate, Acephate", "yield_per_acre": 18},
    "Wheat": {"fertilizer": "Urea, DAP, MOP", "pesticide": "Chlorpyrifos, Malathion", "yield_per_acre": 22},
    "Rice": {"fertilizer": "Urea, Potash, DAP", "pesticide": "Buprofezin, Imidacloprid", "yield_per_acre": 30},
}


try:
    model, le_state, le_season, le_crop = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    st.error(f"❌ Could not load model/encoders: {e}")
    artifacts_loaded = False

if artifacts_loaded:
    states = sorted(list(map(str, le_state.classes_)))
    seasons = sorted(list(map(str, le_season.classes_)))
    state_name = st.selectbox("Select State:", states)
    season_name = st.selectbox("Select Season:", seasons)
    area = st.number_input("Enter Area (in acres):", min_value=1.0, max_value=10000.0, value=5.0, step=0.5)

    if st.button("🔍 Recommend Crops"):
        state_name_clean = state_name.strip().title()
        season_name_clean = season_name.strip().title()

        if state_name_clean not in le_state.classes_.tolist():
            st.error(f"Unknown state: {state_name_clean}")
        elif season_name_clean not in le_season.classes_.tolist():
            st.error(f"Unknown season: {season_name_clean}")
        else:
            s_enc = int(le_state.transform([state_name_clean])[0])
            ss_enc = int(le_season.transform([season_name_clean])[0])

        
            try:
                probs = model.predict_proba([[s_enc, ss_enc]])[0]
                encoded_labels = np.array(model.classes_, dtype=int)
                top_idxs_sorted = np.argsort(probs)[-3:][::-1]
                top_encoded = encoded_labels[top_idxs_sorted]
                top_probs = probs[top_idxs_sorted] * 100
                top_crops = le_crop.inverse_transform(top_encoded)
            except AttributeError:
                pred = model.predict([[s_enc, ss_enc]])[0]
                top_crops = le_crop.inverse_transform([pred])
                top_probs = np.array([100.0])

            # Display results
            st.markdown(f"### 🌾 Top Recommendations for {state_name_clean} ({season_name_clean}) — Area: {area} acres")
            for i, (crop_name, conf) in enumerate(zip(top_crops, top_probs), start=1):
                st.markdown(f"**{i}. {crop_name}**")

                details = crop_details.get(crop_name)
                if details:
                    est_yield = details["yield_per_acre"] * area
                    st.markdown(
                        f"- 💧 **Fertilizer:** {details['fertilizer']}\n"
                        f"- 🐛 **Pesticide:** {details['pesticide']}\n"
                        f"- 🌾 **Estimated Yield:** {est_yield:.1f} quintals (for {area} acres)\n"
                    )
                else:
                    est_yield = 18 * area
                    st.warning(
                        f"💡 General advice for **{crop_name}**:\n\n"
                        f"- 💧 **Fertilizer:** Urea or DAP (General Purpose)\n"
                        f"- 🐛 **Pesticide:** Neem Oil or Bio-Pesticide (General Use)\n"
                        f"- 🌾 **Estimated Yield:** {est_yield:.1f} quintals (approx for {area} acres)\n"
                    )

            # 🌤️ Weather Info
            st.markdown("---")
            st.subheader("🌤️ Current Weather (approx)")
            weather_text = get_weather_text(state_name_clean)
            st.info(weather_text)

# Footer
st.markdown("---")
st.caption("Developed by **Sanskar Gurjar** ✨")
