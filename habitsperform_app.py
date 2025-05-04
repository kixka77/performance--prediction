import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils import shuffle

# Simulated balanced dataset
data = {
    'Study Hours': np.tile([1, 2, 3, 4], 10),
    'Sleep Hours': np.tile([4, 6, 7, 8], 10),
    'Attendance Rate': np.tile([50, 75, 85, 95], 10),
    'Class': ['At Risk'] * 10 + ['Needs Improvement'] * 10 + ['Satisfactory'] * 10 + ['Excellent'] * 10
}
df = pd.DataFrame(data)
df = shuffle(df, random_state=42)

# Encode target
label_map = {'At Risk': 0, 'Needs Improvement': 1, 'Satisfactory': 2, 'Excellent': 3}
df['Target'] = df['Class'].map(label_map)

# Feature & label
X = df[['Study Hours', 'Sleep Hours', 'Attendance Rate']]
y = df['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Inverse map
inverse_label_map = {v: k for k, v in label_map.items()}

# Feedback
feedback_map = {
    'At Risk': 'Seek academic counseling and manage time more effectively.',
    'Needs Improvement': 'Try to study more regularly and improve sleep consistency.',
    'Satisfactory': 'You’re doing okay—keep it up and aim for excellence.',
    'Excellent': 'Great job! Continue with your current habits.'
}

# Streamlit UI
st.title("Academic Performance & Study Habits Predictor")
st.write("Enter your study data to predict your academic standing and receive feedback.")

study_hours = st.slider("Study Hours per Day", 0, 10, 3)
sleep_hours = st.slider("Sleep Hours per Day", 0, 12, 7)
attendance = st.slider("Attendance Rate (%)", 0, 100, 85)

if st.button("Predict"):
    input_data = np.array([[study_hours, sleep_hours, attendance]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prediction_label = inverse_label_map[prediction]

    st.subheader(f"Predicted Category: {prediction_label}")
    st.info(feedback_map[prediction_label])

    st.markdown("### Model Evaluation Metrics")
    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write(f"**Precision**: {precision:.2f}")
    st.write(f"**Recall**: {recall:.2f}")
    st.write(f"**F1 Score**: {f1:.2f}")
