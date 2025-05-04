import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Title
st.title("Student Performance and Study Habits Predictor")
st.write("Enter your real-time study habits and academic data to predict your performance category.")

# Input fields
study_hours = st.number_input("Study Hours per Week", min_value=0.0, max_value=100.0, step=0.5)
attendance = st.slider("Class Attendance (%)", 0, 100, step=1)
assignments_completed = st.slider("Assignments Completed (%)", 0, 100, step=1)
sleep_hours = st.number_input("Average Sleep Hours per Day", min_value=0.0, max_value=24.0, step=0.5)
social_media_hours = st.number_input("Social Media Hours per Day", min_value=0.0, max_value=24.0, step=0.5)
participation = st.slider("Class Participation Level (0=Low, 100=High)", 0, 100, step=1)

# Convert inputs to dataframe
user_input = pd.DataFrame([[
    study_hours, attendance, assignments_completed,
    sleep_hours, social_media_hours, participation
]], columns=['study_hours', 'attendance', 'assignments_completed', 'sleep_hours', 'social_media_hours', 'participation'])

# Sample dataset generation (replace with actual dataset for production)
np.random.seed(42)
sample_data = pd.DataFrame({
    'study_hours': np.random.normal(15, 5, 200),
    'attendance': np.random.uniform(60, 100, 200),
    'assignments_completed': np.random.uniform(50, 100, 200),
    'sleep_hours': np.random.normal(6, 1.5, 200),
    'social_media_hours': np.random.normal(3, 1, 200),
    'participation': np.random.uniform(40, 100, 200),
})
sample_data['category'] = pd.cut(
    0.4*sample_data['study_hours'] +
    0.3*sample_data['attendance'] +
    0.2*sample_data['assignments_completed'] -
    0.3*sample_data['social_media_hours'] +
    0.2*sample_data['participation'],
    bins=[0, 70, 100, 130, 200],
    labels=['At Risk', 'Needs Improvement', 'Satisfactory', 'Excellent']
)

# Train/test split
X = sample_data.drop("category", axis=1)
y = sample_data["category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
user_input_scaled = scaler.transform(user_input)

# Model
model = GradientBoostingClassifier()
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Metrics Display
st.sidebar.header("Model Evaluation Metrics")
st.sidebar.write(f"**Accuracy:** {accuracy:.2f}")
st.sidebar.write(f"**Precision:** {precision:.2f}")
st.sidebar.write(f"**Recall:** {recall:.2f}")
st.sidebar.write(f"**F1 Score:** {f1:.2f}")

# Recommendation function
def get_recommendation(category):
    if category == "Excellent":
        return "Keep up the great work! Continue maintaining strong study habits and good time management."
    elif category == "Satisfactory":
        return "You're doing well, but there’s room for growth. Try increasing your study hours and reducing distractions."
    elif category == "Needs Improvement":
        return "Focus on completing assignments and increasing class engagement. Review difficult subjects regularly."
    elif category == "At Risk":
        return "You're at risk. Seek academic support, attend classes regularly, and reduce social media usage during study time."
    else:
        return "No recommendation available."

# Prediction and recommendation
if st.button("Predict Performance"):
    prediction = model.predict(user_input_scaled)[0]
    st.subheader("Predicted Performance Category:")
    st.success(prediction)

    recommendation = get_recommendation(prediction)
    st.subheader("Feedback and Recommendation:")
    st.info(recommendation)
