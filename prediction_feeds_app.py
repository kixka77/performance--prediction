import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Sample input labels and simulated dataset for training
feature_names = [
    "Study Hours per Day", "Sleep Hours", "Attendance Rate (%)",
    "Quiz Average", "Assignment Score", "Midterm Score", "Final Exam Score"
]

# Simulate dataset
np.random.seed(42)
X = np.random.rand(300, len(feature_names)) * [6, 10, 100, 100, 100, 100, 100]
y = np.random.choice(["Excellent", "Satisfactory", "Needs Improvement", "At Risk"], 300)

# Encode target
label_map = {"Excellent": 0, "Satisfactory": 1, "Needs Improvement": 2, "At Risk": 3}
y_encoded = np.array([label_map[label] for label in y])

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Gradient Boosting Model (used for evaluation purposes)
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_scaled, y_train)

# Deep Neural Network Model
model_dnn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')  # 4 categories
])

model_dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_dnn.fit(X_train_scaled, to_categorical(y_train), epochs=20, batch_size=16, verbose=0)

# Reverse label map for output
reverse_label_map = {v: k for k, v in label_map.items()}

# Feedback system
feedback_dict = {
    "Excellent": "Great job! Keep maintaining your study habits.",
    "Satisfactory": "You're doing well, but there’s room to improve your consistency.",
    "Needs Improvement": "Try organizing your study schedule and focus on weak areas.",
    "At Risk": "Seek help from mentors or join group studies. Adjust your habits as needed."
}

# Streamlit App
st.title("Student Performance & Study Habits Predictor")

st.write("### Enter your details:")

user_input = []
for label in feature_names:
    val = st.number_input(label, min_value=0.0, format="%.2f")
    user_input.append(val)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # DNN prediction
    prediction_probs = model_dnn.predict(input_scaled)
    predicted_class = np.argmax(prediction_probs, axis=1)[0]
    prediction_label = reverse_label_map[predicted_class]

    st.success(f"Predicted Category: **{prediction_label}**")
    st.info(f"Feedback: {feedback_dict[prediction_label]}")
