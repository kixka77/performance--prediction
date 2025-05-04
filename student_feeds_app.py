import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Title
st.title("Student Academic Performance & Study Habits Predictor")

# User Inputs
st.subheader("Enter your data:")
study_time = st.slider("Daily Study Hours", 0, 12, 2)
sleep_time = st.slider("Sleep Hours", 0, 12, 6)
attendance = st.slider("Attendance (%)", 0, 100, 85)
assignments_done = st.slider("Assignments Completed (%)", 0, 100, 80)
extra_activities = st.selectbox("Participates in Extra-curricular Activities?", ['Yes', 'No'])

# Preprocess input
extra_activities_val = 1 if extra_activities == 'Yes' else 0
input_data = pd.DataFrame([[study_time, sleep_time, attendance, assignments_done, extra_activities_val]],
                          columns=['study_time', 'sleep_time', 'attendance', 'assignments_done', 'extra_activities'])

# Simulated dataset for training (you’ll replace this with your own real dataset later)
np.random.seed(0)
X = pd.DataFrame({
    'study_time': np.random.randint(0, 12, 200),
    'sleep_time': np.random.randint(4, 10, 200),
    'attendance': np.random.randint(50, 100, 200),
    'assignments_done': np.random.randint(50, 100, 200),
    'extra_activities': np.random.randint(0, 2, 200)
})
y = pd.cut(
    X['study_time'] * 0.4 + X['attendance'] * 0.3 + X['assignments_done'] * 0.3,
    bins=[0, 40, 60, 80, 100],
    labels=['At Risk', 'Needs Improvement', 'Satisfactory', 'Excellent']
)

# Encode target
y_encoded = y.astype(str)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_input_scaled = scaler.transform(input_data)

# DNN Model
model_dnn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])
model_dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Map labels to numbers
label_map = {'At Risk': 0, 'Needs Improvement': 1, 'Satisfactory': 2, 'Excellent': 3}
y_train_num = y_train.map(label_map)

# Train DNN
model_dnn.fit(X_train_scaled, y_train_num, epochs=20, batch_size=16, verbose=0)

# Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# Predict from both models
dnn_pred = model_dnn.predict(X_input_scaled)
dnn_label = np.argmax(dnn_pred, axis=1)[0]
gb_label = gb_model.predict(input_data)[0]

# Majority Voting
reverse_label_map = {v: k for k, v in label_map.items()}
final_prediction = reverse_label_map.get(dnn_label) if reverse_label_map.get(dnn_label) == gb_label else gb_label

# Evaluation Metrics
y_pred_test = gb_model.predict(X_test)
acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

# Feedback Recommendations
feedback_map = {
    'Excellent': "Keep up the great work! Consider mentoring others or joining enrichment programs.",
    'Satisfactory': "You're doing well. Stay consistent and try scheduling reviews.",
    'Needs Improvement': "Improve focus and time management. Break tasks into smaller, manageable parts.",
    'At Risk': "You're at risk of low performance. Seek academic help and re-evaluate your study habits immediately."
}

# Display Results
st.subheader("Prediction")
st.write(f"**Category:** {final_prediction}")
st.markdown(f"**Recommendation:** {feedback_map.get(final_prediction, 'No feedback available.')}")

# Optional Resources
with st.expander("Study Tips and Academic Support"):
    st.markdown("- Use Pomodoro technique\n- Reduce distractions\n- Attend tutorials\n- Use a planner or study app")

# Show Metrics
st.subheader("Model Evaluation Metrics")
st.write(f"Accuracy: {acc:.2f}")
st.write(f"Precision: {prec:.2f}")
st.write(f"Recall: {rec:.2f}")
st.write(f"F1 Score: {f1:.2f}")
