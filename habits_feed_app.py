import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Define categories and feedback
categories = ['At Risk', 'Needs Improvement', 'Satisfactory', 'Excellent']
feedback = {
    0: "Immediate intervention is recommended. Seek mentoring and improve time management.",
    1: "Focus on consistent study habits and clarify doubts regularly.",
    2: "Good progress! Continue refining your study methods.",
    3: "Excellent performance. Consider tutoring peers or joining academic competitions."
}

# Streamlit UI
st.title("Student Academic Performance and Study Habits Predictor")

study_hours = st.slider("Study Hours per Week", 0, 40, 10)
attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 75)
participation = st.selectbox("Class Participation Level", ["Low", "Medium", "High"])
assignments_score = st.slider("Average Assignment Score (%)", 0, 100, 70)
sleep_hours = st.slider("Average Sleep Hours", 0, 12, 6)

# Map categorical to numerical
participation_dict = {"Low": 0, "Medium": 1, "High": 2}
participation_val = participation_dict[participation]

input_data = np.array([[study_hours, attendance_rate, participation_val, assignments_score, sleep_hours]])

# Simulated training data
np.random.seed(42)
X = np.random.rand(500, 5) * [40, 100, 2, 100, 12]
y = []

for i in range(500):
    score = X[i][0]*0.3 + X[i][1]*0.2 + X[i][2]*10 + X[i][3]*0.3 + X[i][4]*0.2
    if score < 60:
        y.append(0)
    elif score < 75:
        y.append(1)
    elif score < 85:
        y.append(2)
    else:
        y.append(3)

X = np.array(X)
y = np.array(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
input_scaled = scaler.transform(input_data)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    st.subheader(f"Prediction: {categories[prediction]}")
    st.write(f"Feedback: {feedback[prediction]}")
