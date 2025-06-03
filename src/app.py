import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
from src.data_generation import generate_food_dataset
from src.model_training import train_model, predict_glucose
from src.utils import setup_logging

# Set up logging
logger = setup_logging()

# Streamlit page configuration
st.set_page_config(page_title="Food Glucose Predictor", layout="wide")

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.vectorizer = None
    st.session_state.df = None

def load_data_and_model():
    """Load or generate dataset and model."""
    try:
        # Load or generate dataset
        if os.path.exists("../food_carbohydrate_dataset.csv"):
            st.session_state.df = pd.read_csv("../food_carbohydrate_dataset.csv")
            logger.info("Loaded existing dataset.")
        else:
            st.session_state.df = generate_food_dataset()
            st.session_state.df.to_csv("../food_carbohydrate_dataset.csv", index=False)
            logger.info("Generated and saved new dataset.")
        
        # Load or train model
        if os.path.exists("../food_glucose_model.pkl") and os.path.exists("../food_vectorizer.pkl"):
            with open("../food_glucose_model.pkl", "rb") as f:
                st.session_state.model = pickle.load(f)
            with open("../food_vectorizer.pkl", "rb") as f:
                st.session_state.vectorizer = pickle.load(f)
            logger.info("Loaded existing model and vectorizer.")
        else:
            st.session_state.model, st.session_state.vectorizer = train_model(st.session_state.df)
            with open("../food_glucose_model.pkl", "wb") as f:
                pickle.dump(st.session_state.model, f)
            with open("../food_vectorizer.pkl", "wb") as f:
                pickle.dump(st.session_state.vectorizer, f)
            logger.info("Trained and saved new model and vectorizer.")
    
    except Exception as e:
        logger.error(f"Error loading data/model: {e}")
        st.error(f"Error loading data/model: {e}")

def get_diabetic_recommendation(glucose_content):
    """
    Determine if a food is recommended for diabetic patients based on glucose content.
    Args:
        glucose_content (float): Predicted glucose content (g/100g).
    Returns:
        tuple: (recommendation text, color)
    """
    if glucose_content < 10:
        return "Recommended: Low glucose content, safe for diabetic patients.", "green"
    elif 10 <= glucose_content <= 20:
        return "Caution: Moderate glucose content, consume in moderation for diabetic patients.", "orange"
    else:
        return "Not Recommended: High glucose content, may cause blood sugar spikes.", "red"

# Main app
def main():
    st.title("Food Glucose Predictor")
    st.markdown("Explore nutritional data and predict glucose content for Ethiopian and European foods, with recommendations for diabetic patients.")
    
    # Load data and model
    if st.button("Load/Generate Data and Model"):
        with st.spinner("Loading data and model..."):
            load_data_and_model()
        st.success("Data and model loaded successfully!")
    
    if st.session_state.df is not None:
        # Sidebar for navigation
        st.sidebar.header("Navigation")
        page = st.sidebar.radio("Select a page:", ["Dataset Explorer", "Glucose Predictor", "Data Visualizations"])
        
        # Dataset Explorer
        if page == "Dataset Explorer":
            st.header("Dataset Explorer")
            st.write("Browse the generated dataset of foods and their nutritional content.")
            st.dataframe(st.session_state.df)
            
            # Download dataset
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="Download Dataset as CSV",
                data=csv,
                file_name="food_carbohydrate_dataset.csv",
                mime="text/csv"
            )
        
        # Glucose Predictor
        elif page == "Glucose Predictor":
            st.header("Glucose Content Predictor")
            st.write("Enter a food name to predict its glucose content (g/100g) and check if it's recommended for diabetic patients.")
            food_name = st.text_input("Food Name", placeholder="e.g., Injera, Pasta")
            
            if st.button("Predict"):
                if food_name and st.session_state.model and st.session_state.vectorizer:
                    try:
                        prediction = predict_glucose(food_name, st.session_state.model, st.session_state.vectorizer)
                        recommendation, color = get_diabetic_recommendation(prediction)
                        st.success(f"Predicted glucose content for '{food_name}': **{prediction} g/100g**")
                        st.markdown(f"<p style='color:{color};'>{recommendation}</p>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error predicting: {e}")
                else:
                    st.error("Please enter a food name and ensure model is loaded.")
        
        # Data Visualizations
        elif page == "Data Visualizations":
            st.header("Data Visualizations")
            st.write("Explore nutritional data distributions.")
            
            # Glucose distribution by category
            fig = px.histogram(
                st.session_state.df,
                x="Glucose_g_per_100g",
                color="Category",
                title="Glucose Content Distribution by Food Category",
                labels={"Glucose_g_per_100g": "Glucose (g/100g)"},
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Nutritional comparison
            nutrient = st.selectbox("Select Nutrient to Visualize", ["Glucose_g_per_100g", "Carbohydrate_g_per_100g", "Glycemic_Index", "Calories_kcal_per_100g", "Protein_g_per_100g", "Fat_g_per_100g"])
            fig2 = px.box(
                st.session_state.df,
                x="Category",
                y=nutrient,
                title=f"{nutrient.replace('_', ' ').title()} by Food Category",
                labels={nutrient: nutrient.replace('_', ' ').title()}
            )
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()