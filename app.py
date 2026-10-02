import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('StudentPerformanceFactors.csv.pkl')

encoder = joblib.load('StudentPerformanceFactors.csv_encoder.pkl')

feature_info = joblib.load('StudentPerformanceFactors.csv_info.pkl')

# Load trained model
model = joblib.load(
    r'C:\Users\ADMIN\Startup Project\StudentPerformanceFactors.csv.pkl'
)

# Load encoder
encoder = joblib.load(
    r'C:\Users\ADMIN\Startup Project\StudentPerformanceFactors.csv_encoder.pkl'
)

# Load feature information
feature_info = joblib.load(
    r'C:\Users\ADMIN\Startup Project\StudentPerformanceFactors.csv_info.pkl'
)

# App title
st.title("🎓 Student Performance Prediction Dashboard")
st.write("Enter student details to predict the exam score.")

st.header("📚 Student Academic Details")

hours_studied = st.number_input(
    "Hours Studied",
    min_value=1,
    max_value=50,
    value=20
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=85
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=1,
    max_value=24,
    value=7
)

previous_scores = st.number_input(
    "Previous Scores",
    min_value=0,
    max_value=100,
    value=75
)

tutoring_sessions = st.number_input(
    "Tutoring Sessions",
    min_value=0,
    max_value=20,
    value=2
)

physical_activity = st.number_input(
    "Physical Activity",
    min_value=0,
    max_value=20,
    value=3
)

st.header("👨‍👩‍👧 Student Personal & Family Details")

parental_involvement = st.selectbox(
    "Parental Involvement",
    ["High", "Low", "Medium"]
)

access_to_resources = st.selectbox(
    "Access to Resources",
    ["High", "Low", "Medium"]
)

extracurricular_activities = st.selectbox(
    "Extracurricular Activities",
    ["No", "Yes"]
)

motivation_level = st.selectbox(
    "Motivation Level",
    ["High", "Low", "Medium"]
)

internet_access = st.selectbox(
    "Internet Access",
    ["No", "Yes"]
)

family_income = st.selectbox(
    "Family Income",
    ["High", "Low", "Medium"]
)

teacher_quality = st.selectbox(
    "Teacher Quality",
    ["High", "Low", "Medium"]
)

school_type = st.selectbox(
    "School Type",
    ["Private", "Public"]
)

peer_influence = st.selectbox(
    "Peer Influence",
    ["Negative", "Neutral", "Positive"]
)

learning_disabilities = st.selectbox(
    "Learning Disabilities",
    ["No", "Yes"]
)

parental_education_level = st.selectbox(
    "Parental Education Level",
    ["College", "High School", "Postgraduate"]
)

distance_from_home = st.selectbox(
    "Distance from Home",
    ["Far", "Moderate", "Near"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

if st.button("🔮 Predict Exam Score"):

    student_data = {
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Sleep_Hours': sleep_hours,
        'Previous_Scores': previous_scores,
        'Tutoring_Sessions': tutoring_sessions,
        'Physical_Activity': physical_activity,

        'Parental_Involvement': parental_involvement,
        'Access_to_Resources': access_to_resources,
        'Extracurricular_Activities': extracurricular_activities,
        'Motivation_Level': motivation_level,
        'Internet_Access': internet_access,
        'Family_Income': family_income,
        'Teacher_Quality': teacher_quality,
        'School_Type': school_type,
        'Peer_Influence': peer_influence,
        'Learning_Disabilities': learning_disabilities,
        'Parental_Education_Level': parental_education_level,
        'Distance_from_Home': distance_from_home,
        'Gender': gender
    }

    student_df = pd.DataFrame([student_data])

    # Encode categorical features
    student_categorical = encoder.transform(
        student_df[feature_info['categorical_features']]
    )

    # Numerical features
    student_numerical = student_df[
        feature_info['numeric_features']
    ].to_numpy()

    # Combine all features
    student_final = np.hstack([
        student_numerical,
        student_categorical
    ])

    # Prediction
    predicted_score = model.predict(student_final)[0]

    st.subheader("🎯 Predicted Exam Score")
    st.write(f"{predicted_score:.2f}")
