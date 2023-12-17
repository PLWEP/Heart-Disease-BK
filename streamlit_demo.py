import streamlit as st
import pandas as pd
import pickle
import pickle

model_filename = 'best_model.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)


def main():
    st.title('Heart Disease Prediction')

    # Feature 1
    age = st.slider('Age', 0, 100, 50)

    # Feature 2
    sex_options = ['Male', 'Female']
    sex = st.selectbox('Sex', sex_options)
    sex_num = sex_options.index(sex)

    # Feature 3
    cp_options = ['Typical Angina', 'Atypical Angina',
                  'Non-anginal Pain', 'Asymptomatic']
    cp = st.selectbox('Chest Pain Type', cp_options)
    cp_num = cp_options.index(cp)

    # Feature 4
    trestbps = st.slider('Resting Blood Pressure (mm/Hg)', 90, 200, 120)

    # Feature 5
    chol = st.slider('Serum Cholestoral (mg/dl)', 100, 600, 250)

    # Feature 6
    fbs_options = ['False', 'True']
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', fbs_options)
    fbs_num = fbs_options.index(fbs)

    # Feature 6
    restecg_options = ['Normal', 'ST-T Abnormality',
                       'Left Ventricular Hypertrophy']
    restecg = st.selectbox(
        'Resting Electrocardiographic Results', restecg_options)
    restecg_num = restecg_options.index(restecg)

    # Feature 7
    thalach = st.slider('Maximum Heart Rate Achieved', 70, 220, 150)

    # Feature 8
    exang_options = ['No', 'Yes']
    exang = st.selectbox('Exercise Induced Angina', exang_options)
    exang_num = exang_options.index(exang)

    # Feature 9
    oldpeak = st.slider(
        'ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)

    # Feature 10
    slope_options = ['Upsloping', 'Flat', 'Downsloping']
    slope = st.selectbox(
        'Slope of the Peak Exercise ST Segment', slope_options)
    slope_num = slope_options.index(slope)

    # Feature 10
    ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 3, 1)

    # Feature 11
    thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
    thal = st.selectbox('Thalassemia', thal_options)
    thal_num = thal_options.index(thal)

    if st.button('Predict'):
        user_input = pd.DataFrame(data={
            'age': [age],
            'sex': [sex_num],
            'cp': [cp_num],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs_num],
            'restecg': [restecg_num],
            'thalach': [thalach],
            'exang': [exang_num],
            'oldpeak': [oldpeak],
            'slope': [slope_num],
            'ca': [ca],
            'thal': [thal_num]
        })
        prediction = model.predict(user_input.values)
        prediction_proba = model.predict_proba(user_input.values)

        if prediction[0] == 1:
            bg_color = 'red'
            prediction_result = 'Positive'
        else:
            bg_color = 'green'
            prediction_result = 'Negative'

        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

        st.markdown(
            f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction: {prediction_result}<br>Confidence: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
