import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define feedback based on prediction
def get_feedback(category):
    feedback_map = {
        "At Risk": "Seek academic counseling, manage time better, and consult your instructors regularly.",
        "Needs Improvement": "Improve study habits and consider joining a peer study group.",
        "Satisfactory": "You are doing well, but there's still room for consistency and deeper learning.",
        "Excellent": "Outstanding! Keep up the great work and consider mentoring others."
    }
    return feedback_map.get(category, "No feedback available.")

# Sample dataset placeholder (replace with your actual CSV/DB data source if needed)
@st.cache_data
def load_data():
    np.random.seed(42)
    X = np.random.rand(300, 6)  # 6 features
    y = np.random.choice(["At Risk", "Needs Improvement", "Satisfactory", "Excellent"], 300)
    return pd.DataFrame(X, columns=['Quiz Score', 'Assignment Score', 'Study Hours', 'Sleep Hours', 'Attendance', 'Participation']), pd.Series(y)

# Load and encode data
X, y = load_data()
y_encoded = pd.Categorical(y).codes  # Convert categories to 0–3 integers
labels = ['At Risk', 'Needs Improvement', 'Satisfactory', 'Excellent']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Ensure y_train is int32 for TensorFlow
y_train = np.array(y_train).astype("int32")

# Build DNN model
model_dnn = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(4, activation='softmax')
])
model_dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_dnn.fit(X_train_scaled, y_train, epochs=20, batch_size=16, verbose=0)

# Streamlit UI
st.title("Student Performance and Study Habit Predictor")

# Input form
quiz = st.slider("Quiz Score (0-100)", 0, 100, 75)
assign = st.slider("Assignment Score (0-100)", 0, 100, 80)
study = st.slider("Study Hours per Day", 0, 12, 3)
sleep = st.slider("Sleep Hours per Day", 0, 12, 6)
attend = st.slider("Attendance (%)", 0, 100, 90)
particip = st.slider("Class Participation (0-100)", 0, 100, 85)

# Predict
if st.button("Predict"):
    input_data = np.array([[quiz, assign, study, sleep, attend, particip]])
    input_scaled = scaler.transform(input_data)
    prediction = model_dnn.predict(input_scaled)
    category_idx = np.argmax(prediction)
    category = labels[category_idx]
    st.subheader(f"Prediction: {category}")
    st.info(get_feedback(category))

    # Optional: Display classification report
    y_pred = np.argmax(model_dnn.predict(X_train_scaled), axis=1)
    report = classification_report(y_train, y_pred, target_names=labels, output_dict=True)
    st.write("Model Evaluation Metrics:")
    st.json({
        "Accuracy": round(report['accuracy'], 2),
        "Precision": {k: round(v['precision'], 2) for k, v in report.items() if k in labels},
        "Recall": {k: round(v['recall'], 2) for k, v in report.items() if k in labels},
        "F1-score": {k: round(v['f1-score'], 2) for k, v in report.items() if k in labels}
    })
