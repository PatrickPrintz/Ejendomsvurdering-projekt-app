import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load data
shap_values = pd.read_csv("shap_values.csv")
shap_values = shap_values.to_numpy()
X_test_shh = pd.read_csv("X_test.csv")
X_test_df = pd.read_csv("X_test_df.csv")
explainer_expected_value = 2678679.5

# Streamlit app
st.title('SHAP Waterfall Plot App')
max_value=len(shap_values)
# Number input for 'nummer'
nummer = st.number_input('Vælg et tilfældigt nummer mellem 0 og 19.570:', min_value=0, max_value=len(shap_values)-1, value=0, step=1)

# Button to generate and show the plot
if st.button('Generer forklarings plot af vurdering'):
    st.text(f"""
    Forklaringen af vurderingen er baseret på en SHAP-waterfall plot.
            Den indtastede bolig er et {X_test_df.iloc[nummer]['anvendelse']} på {X_test_df.iloc[nummer]['area']} m2, placeret i {X_test_df.iloc[nummer]['Region']}.
            HELLO!!!!
            """)
    # SHAP-waterfall plot
    plt.figure(figsize=(5, 5))  
    fig = shap.plots.waterfall(shap.Explanation(values=shap_values[nummer],
                                     base_values=explainer_expected_value,
                                     data=X_test_shh.iloc[nummer]),
                                     max_display=25)
    st.pyplot(fig)


