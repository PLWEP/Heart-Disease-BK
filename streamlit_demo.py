import pickle
import time
import pandas as pd
import streamlit as st


model = pickle.load(open('best_model.pkl', 'rb'))
model_info = pickle.load(open('model_info.pkl', 'rb'))
scaler_model = pickle.load(open('scaler_model.pkl', 'rb'))
data_final = model_info['parameter']
accuracy = model_info['accuracy']

# mengatur title web browser
st.set_page_config(
    page_title="Hungarian Heart Disease",
    page_icon=":heart:"
)

# judul webpage
st.title("Hungarian Heart Disease")

st.image("https://img.freepik.com/free-vector/doctor-with-stethoscope-listening-huge-heart-beat-ischemic-heart-disease_335657-4397.jpg?w=996&t=st=1704173675~exp=1704174275~hmac=0f712a91cf89a887e75df001cbe07e2b56b7159dd56890783fc07a5d43841b76", width=500)

# _ : italic, ** : bold
st.write(
    f"**_Model's Accuracy_** : :green[**{accuracy}**]%")

tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])


with tab1:
    st.sidebar.header("**User Input** Sidebar")

    age = st.sidebar.number_input(
        label="**Age**", min_value=int(data_final['age'].min()), max_value=int(data_final['age'].max()), step=1)
    st.sidebar.write(
        f"Min value: {data_final['age'].min()}, Max value: {data_final['age'].max()}")

    sex_cb = st.sidebar.selectbox(
        label="**Sex**", options=["Male", "Female"])
    if sex_cb == "Male":
        sex = 1
    elif sex_cb == "Female":
        sex = 0
    # --Value 0 : Female
    # --Value 1 : Male

    cp_sb = st.sidebar.selectbox(label="**Chest pain type**", options=[
                                 "Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
    if cp_sb == "Typical angina":
        cp = 1
    elif cp_sb == "Atypical angina":
        cp = 2
    elif cp_sb == "Non-anginal pain":
        cp = 3
    elif cp_sb == "Asymptomatic":
        cp = 4
    # --Value 1 : typical angina
    # --Value 2 : atypical angina
    # --Value 3 : non-anginal pain
    # --Value 4 : asymptomatic

    trestbps = st.sidebar.number_input(label="**Resting blood pressure** (in mm Hg on admission to the hospital)",
                                       min_value=data_final['trestbps'].min(), max_value=data_final['trestbps'].max())
    st.sidebar.write(
        f"Min value: {data_final['trestbps'].min()}, Max value: {data_final['trestbps'].max()}")

    chol = st.sidebar.number_input(label="**Serum cholestoral** (in mg/dl)",
                                   min_value=data_final['chol'].min(), max_value=data_final['chol'].max())
    st.sidebar.write(
        f"Min value: {data_final['chol'].min()}, Max value: {data_final['chol'].max()}")

    fbs_sb = st.sidebar.selectbox(
        label="**Fasting blood sugar > 120 mg/dl?**", options=["False", "True"])
    if fbs_sb == "False":
        fbs = 0
    elif fbs_sb == "True":
        fbs = 1
    # --Value 0 : False
    # --Value 1 : True

    restecg_sb = st.sidebar.selectbox(label="**Resting electrocardiographic result**", options=[
                                      "Normal", "having ST-T wave abnormality", "Showing lwft ventricular hypertrophy"])
    if restecg_sb == "Normal":
        restecg = 0
    elif restecg_sb == "having ST-T wave abnormality":
        restcg = 1
    elif restecg_sb == "Showing left ventricular hypertrophy":
        restecg = 2
    # --Value 0 : Normal
    # --Value 1 : Having ST-T wave abnormality (T wave inverssion and/or ST elevation or depression of > 0.05 mV)
    # --Value 2 : Showing probable or definite lef ventricular hypertrophy by Estes' criteria

    thalach = st.sidebar.number_input(
        label="**Maximum heart rate achieved**", min_value=data_final['thalach'].min(), max_value=data_final['thalach'].max())
    st.sidebar.write(
        f"Min value: {data_final['thalach'].min()}, Max value: {data_final['thalach'].max()}")

    exang_sb = st.sidebar.selectbox(
        label="**Exercise induced angina?**", options=["No", "Yes"])
    if exang_sb == "No":
        exang = 0
    elif exang_sb == "Yes":
        exang = 1
    # --Value 0 : No
    # --Value 1 : Yes

    oldpeak = st.sidebar.number_input(label="**ST depression induced by exercise relativve to rest**",
                                      min_value=data_final['oldpeak'].min(), max_value=data_final['oldpeak'].max())
    st.sidebar.write(
        f"Min value: {data_final['oldpeak'].min()}, Max value: {data_final['oldpeak'].max()}")

    data = {
        'Age': float(age),
        'Sex': sex_cb,
        'Chest pain type': cp_sb,
        'RPB': f"{trestbps} mm Hg",
        'Serum Cholestoral': f"{chol} mg/dl",
        'FBS > 120 mg/dl?': fbs_sb,
        'Maximum heart rate': thalach,
        'Exercise induced angina?': exang_sb,
        'ST depression': oldpeak,
    }
    preview_data_input = pd.DataFrame(data, index=['input'])

    st.header("User Input as Dataframe")
    st.dataframe(preview_data_input.iloc[:, :6])
    st.dataframe(preview_data_input.iloc[:, 6:])

    result = ":violet[-]"

    predict_btn = st.button("**Predict**", type="primary")

    if predict_btn:
        inputs = [[float(age), sex, cp, trestbps, chol,
                   fbs, restecg, thalach, exang, oldpeak]]
        inputs = scaler_model.transform(inputs)
        prediction = model.predict(inputs)[0]

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        if prediction == 0:
            result = ":green[**Healthy**]"
        elif prediction == 1:
            result = ":orange[**Heart disease level 1**]"
        elif prediction == 2:
            result = ":orange[**Heart disease level 2**]"
        elif prediction == 3:
            result = ":red[**Heart disease level 3**]"
        elif prediction == 4:
            result = ":red[**heart disease level 4**]"

        st.subheader("Prediction :")
        st.subheader(result)

with tab2:
    st.header("Predict multiple data :")

    sample_csv = data_final.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

    st.download_button("Download CSV Example", data=sample_csv,
                       file_name='sample_heart_disease_parameteers.csv', mime='text/csv')

    file_uploaded = st.file_uploader("UPload a CSV file", type='csv')

    if file_uploaded:
        uploaded_df = pd.read_csv(file_uploaded)
        uploaded_df_scaler = scaler_model.transform(uploaded_df)
        prediction_arr = model.predict(uploaded_df_scaler)

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 70):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)

        result_arr = []

        for prediction in prediction_arr:
            if prediction == 0:
                result = "Healthy"
            elif prediction == 1:
                result = "Heart disease level 1"
            elif prediction == 2:
                result = "Heart disease level 2"
            elif prediction == 3:
                result = "Heart disease level 3"
            elif prediction == 4:
                result = "Heart disease level 4"
            result_arr.append(result)

        uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

        for i in range(70, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(uploaded_result)
        with col2:
            st.dataframe(uploaded_df)
