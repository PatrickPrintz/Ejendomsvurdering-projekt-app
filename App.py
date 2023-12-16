import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load data
shap_values = pd.read_csv("shap_values.csv")
X_test_shh = pd.read_csv("X_test.csv")
explainer_expected_value = 2678679.5
shap_values.shape
# Streamlit app
st.title('SHAP Waterfall Plot App')

# Number input for 'nummer'
nummer = st.number_input('Enter a number:', min_value=0, max_value=len(shap_values)-1, value=0, step=1)

# Button to generate and show the plot
if st.button('Generate Waterfall Plot'):
    # SHAP-waterfall plot
    plt.figure(figsize=(5, 5))  
    fig = shap.plots.waterfall(shap.Explanation(values=shap_values[nummer],
                                     base_values=explainer_expected_value,
                                     data=X_test_shh.iloc[nummer]))
    st.pyplot(fig)


