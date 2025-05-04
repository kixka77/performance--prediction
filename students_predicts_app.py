import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Streamlit title
st.title("Student Academic Performance and Study Habit Predictor")

# Input form
st.header("Enter Student Information")
hours_studied = st.number_input("Hours Studied Per Day", min_value=0.0, max_value=24.0, step=0.5)
attendance = st.slider("Attendance Rate (%)", min_value=0, max_value=100, step=1)
sleep_hours = st.slider("Sleep Hours Per Night", min_value=0, max_value=12, step=1)
extra_activities = st.selectbox("Participates in Extra Activities", ["Yes", "No"])
stress_level = st.slider("Stress Level (1=Low, 5=High)", min_value=1, max_value=5)

# Prepare input
input_data = pd.DataFrame({
    'Hours_Studied': [hours_studied],
    'Attendance': [attendance],
    'Sleep_Hours': [sleep_hours],
    'Extra_Activities': [1 if extra_activities == "Yes" else 0],
    'Stress_Level': [stress_level]
})

# Sample training dataset (you should replace this with real collected data in CSV or database)
df = pd.DataFrame({
    'Hours_Studied': np.random.uniform(0, 6, 100),
    'Attendance': np.random.randint(50, 100, 100),
    'Sleep_Hours': np.random.randint(4, 9, 100),
    'Extra_Activities': np.random.randint(0, 2, 100),
    'Stress_Level': np.random.randint(1, 6, 100),
    'Performance': np.random.choice(['At Risk', 'Needs Improvement', 'Satisfactory', 'Excellent'], 100)
})

# Split features and labels
X = df.drop('Performance', axis=1)
y = df['Performance']

# Label encode
label_map = {'At Risk': 0, 'Needs Improvement': 1, 'Satisfactory': 2, 'Excellent': 3}
y_encoded = y.map(label_map)
mask = ~y_encoded.isna()
X = X[mask]
y_encoded = y_encoded[mask].astype(int)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_input_scaled = scaler.transform(input_data)

# Gradient Boosting Model
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_scaled, y_train)

# DNN Model
model_dnn = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(4, activation='softmax')
])
model_dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_dnn.fit(X_train_scaled, y_train, epochs=20, batch_size=16, verbose=0)

# Predictions
gb_pred = gb_model.predict(X_input_scaled)
dnn_pred = np.argmax(model_dnn.predict(X_input_scaled), axis=1)

# Combine results (you can customize this logic)
final_prediction = int(np.round((gb_pred[0] + dnn_pred[0]) / 2))

# Map back to label
reverse_map = {v: k for k, v in label_map.items()}
predicted_label = reverse_map.get(final_prediction, "Unknown")

# Feedback Recommendation
feedback_dict = {
    'Excellent': "Great job! Keep up the good habits.",
    'Satisfactory': "Doing well, but there's still room to push further.",
    'Needs Improvement': "Focus more on consistency and time management.",
    'At Risk': "Seek support from peers or advisors and reduce stress levels."
}
feedback = feedback_dict.get(predicted_label, "No feedback available.")

# Display
st.subheader("Prediction Result")
st.write(f"**Category:** {predicted_label}")
st.write(f"**Feedback:** {feedback}")

# Evaluation metrics (optional)
y_pred_gb = gb_model.predict(X_test)
st.subheader("Model Evaluation (Gradient Boosting)")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.2f}")
st.write(f"F1 Score: {f1_score(y_test, y_pred_gb, average='weighted'):.2f}")
st.write(f"Precision: {precision_score(y_test, y_pred_gb, average='weighted'):.2f}")
st.write(f"Recall: {recall_score(y_test, y_pred_gb, average='weighted'):.2f}")
