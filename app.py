import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

#Import necessary dataset and model
data = pd.read_csv('phl_schp_deped_clean.csv', encoding='ISO-8859-1')
with open("rf_tk.pkl", "rb") as f:
    model = pickle.load(f)

#Define helper functions
def manual_ohe(data, ref_df, column):
    col = ref_df[column].unique()
    for val in col:
        if val not in data[column]:
            data[f"{column}_{val}"] = 0
        else:
            data[f"{column}_{val}"] = 1
    data.drop(column, inplace=True, axis=1)
    return data
st.set_page_config(layout="wide")
st.title("School Type Recommender Engine")
st.subheader("A project by Jamie, Ace, and Martel of DSF11 Eskwelabs")
inp, res = st.columns(2)

with inp.form("New Input"):
    input_data = {}

    input_data["region"] = [st.selectbox("Region", data["region"].unique())]
    input_data["province"] = [st.selectbox("Province", data["province"].unique())]
    input_data["legislative"] = [st.selectbox("Legislative", data["legislative"].unique())]
    input_data["division"] = [st.selectbox("Division", data["division"].unique())]
    input_data["total_enrollees"] = [st.number_input("Total Enrollees", min_value=1)]
    input_data["total_instructors"] = [st.number_input("Total Instructors", min_value=1)]
    input_data["poverty_incidence_among_families"] = [st.number_input("Poverty Incidence Among Families", min_value=0.0, format="%.2f")]
    input_data["population_as_of_may_2020"] = [st.number_input("Population", min_value=1)]
    input_data["unemployment_rate_per_region"] = [st.number_input("Unemployment Rate", min_value=0.0, format="%.2f")]

    submit = st.form_submit_button("SUBMIT")

    if submit:
        df = pd.DataFrame.from_dict(data=input_data)
        
        #Preprocessing
        for col in ["region", "province", "legislative", "division"]:
            df = manual_ohe(df, data, col)

        scaler = StandardScaler()
        scaler = scaler.fit(data['population_as_of_may_2020'].values.reshape(-1, 1))
        df["population_as_of_may_2020"] = scaler.transform(df['population_as_of_may_2020'].values.reshape(-1, 1))
        
        #Generate predictions
        result = model.predict(df)


res.info("See the recommendation here after submitting new data")
if 'result' in locals():
    #Generate recommendation spiel
    if result == '0':
        result = "Annex or Extension School"
    else:
        result = "Lone School"
    res.write(f"""
    For {input_data["region"]} with {input_data["total_instructors"]} total instructors and {input_data["total_enrollees"]} total enrollees,
    a {result} type school is recommended.
        """)