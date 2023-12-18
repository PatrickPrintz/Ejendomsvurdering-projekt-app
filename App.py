import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load data
shap_values = pd.read_csv("shap_values.csv")
shap_values = shap_values.to_numpy()
X_test_shh = pd.read_csv("X_test.csv")
X_test_df = pd.read_csv("X_test_df.csv")

@st.cache_data
def load_data():
    X_train_df = pd.read_csv("X_train.csv")
    return X_train_df

X_train_df = load_data()


energylabel_reversed_map = {7: 'A', 6: 'B', 5: 'C', 4: 'D', 3: 'E', 2: 'F', 1: 'G'}
X_test_df['Energylabel'] = X_test_df['Energylabel'].map(energylabel_reversed_map)
explainer_expected_value = 2678679.5

# Streamlit app
st.title('Forklaring af individuelle vurderinger')
st.write("Denne app har til formål illustrativt at vise hvordan en offentlig hjemmeside, kan opbygges, til at forklare individuelle vurderinger.")

st.write("Formålet er at kunne give en letforståelig måde at forstå enkelte vurderinger.")

st.subheader("Dette er modellen til vurderingerne trænet på")
st.write("I denne sektion vil der være en række af plots, som beskriver det data, som modellen er trænet på.")
st.write("Det tager lidt tid at loade data hver gang, pga. størrelsen af data og computerens ydeevne.")

columns_to_plot = ['None', 'varmesinstallation', 'vægmateriale', 'tagkode', 'Energylabel', 'price',
       'build_year', 'Region', 'grund_str']

# Dropdown menu for selecting a column
selected_column = st.selectbox("Select a column:", columns_to_plot)

# Plotting logic based on the selected column
if selected_column in ['price', 'build_year', 'grund_str']:
    # Histogram for numeric columns
    plt.figure(figsize=(8, 6))
    sns.histplot(X_train_df[selected_column], kde=True)
    plt.title(f'Fordeling af {selected_column}')
    plt.xlabel(selected_column)
    plt.ylabel('Frekvens')
    st.pyplot()
    
elif selected_column == 'None':
    st.write("Vælg en feature fra dropdown menuen for at se data.")

elif selected_column in ['varmesinstallation', 'vægmateriale', 'tagkode', 'Energylabel','Region']:
    plt.figure(figsize=(8, 6))
    sns.countplot(X_train_df[selected_column])
    plt.title(f'Fordeling af {selected_column}')
    plt.xlabel(selected_column)
    plt.ylabel('Antal')
    st.pyplot()













st.subheader("Dette er modellen til vurderingerne trænet på")
st.write("""
Denne sektion vil give et indblik i de enkelte vurderinger af rækkehuse. Vælg et tilfældigt nummer i nedenstående slider,
         Og tryk for knappen for at generere et forklarings plot af vurderingen. Bemærk at grundet OneHotEncoding er der mangle features i figuren.
         Ligeledes vil der af de værdier som fremgår af plottet være benyttet skaleringsmetoder. Læs derfor i stedet teksten for rigtig fakta om enkelte bolig.
""")
nummer = st.number_input('Vælg et tilfældigt nummer mellem 0 og 19.570:', min_value=0, max_value=len(shap_values)-1, value=0, step=1)

# Button to generate and show the plot
if st.button('Generer forklarings plot af vurdering'):
    if X_test_df.iloc[nummer]['ombygaar'] == 0:
        ombyg = "der er ikke registreret nogen ombygninger"
    else:
        ombyg = f"der er registreret en ombygning i år {X_test_df.iloc[nummer]['ombygaar']}"

    st.write(f"""
    Forklaringen af vurderingen er baseret på en SHAP-waterfall plot.
            Den indtastede bolig er et {X_test_df.iloc[nummer]['anvendelse']} på {X_test_df.iloc[nummer]['area']} m2, placeret i {X_test_df.iloc[nummer]['Region']}, med {X_test_df.iloc[nummer]['grund_str']} m2 grund.
            Boligen består af i alt {X_test_df.iloc[nummer]['badtoi']} badeværelser og toiletter og {X_test_df.iloc[nummer]['rooms']} værelser.
            Boligen er opført i {X_test_df.iloc[nummer]['build_year']} og {ombyg}. Boligens primær varmekilde er {X_test_df.iloc[nummer]['varmesinstallation']} og har en energimærkning på {X_test_df.iloc[nummer]['Energylabel']}.
            Boligens ydervæg er lavet af {X_test_df.iloc[nummer]['vægmateriale']} og taget består af {X_test_df.iloc[nummer]['tagkode']}.
            Boligen er placeret {X_test_df.iloc[nummer]['dist_coast'].round(2)} km. fra kysten, {X_test_df.iloc[nummer]['dist_highway'].round(2)} km. fra nærmeste motorvej, 
            {X_test_df.iloc[nummer]['dist_railroads'].round(2)} km. fra nærmeste jernbane, {X_test_df.iloc[nummer]['dist_airports'].round(2)} km. fra nærmeste lufthavn.
            Nærmeste skole er {X_test_df.iloc[nummer]['dist_school'].round(2)} km. væk, nærmeste universitet er {X_test_df.iloc[nummer]['dist_uni'].round(2)} km. væk og nærmeste insitution eller børnehave er {X_test_df.iloc[nummer]['dist_kindergarden'].round(2)} km. væk.
            Boligen er ligeledes placeret tæt på naturen, hvor der er {X_test_df.iloc[nummer]['dist_forests'].round(2)} km. til nærmeste skov og {X_test_df.iloc[nummer]['dist_waterlines'].round(2)} km. til nærmeste sø eller å.
            Naboområdets pris ligger på omstrent {X_test_df.iloc[nummer]['weighted_price'].round(0)} kr. vægtet efter afstanden til boligen.
            Boliger er placeret ved {X_test_df.iloc[nummer]['longitude']} længdegrad og {X_test_df.iloc[nummer]['latitude']} breddegrad.
            Nedenfor fremgår en forklaring på modellens vurdering af boligen.
            """)
    st.write(f"Af Nedenstående figure, fremgår gennemsnitsprisen E(f(x)) = {explainer_expected_value} kr. for alle boliger i test sættet. Hertil fremgår der tillæg eller fradrag for de forskellige features, som er med til at påvirke prisen på boligen.")
    # SHAP-waterfall plot
    plt.figure(figsize=(5, 5))  
    fig = shap.plots.waterfall(shap.Explanation(values=shap_values[nummer],
                                     base_values=explainer_expected_value,
                                     data=X_test_shh.iloc[nummer]),
                                     max_display=10)
    st.pyplot(fig)

    st.write(f"""
    Boligen er oprindeligt solgt for {X_test_df.iloc[nummer]['price'].round()} kr. ved sidste salg.""")
    vurdering = explainer_expected_value + shap_values[nummer].sum()
    st.write(f"Modellens endelige vurdering lyder på: {vurdering.round()} kr.")
    st.write(f"Modellens vurdering afviger derfor med {((vurdering.round() - X_test_df.iloc[nummer]['price'].round())/X_test_df.iloc[nummer]['price'].round() *100).round()} pct. fra den sidste salgspris.")


