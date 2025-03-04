import streamlit as st
import face_recognition
import cv2
import numpy as np
import pickle
import mysql.connector
import time
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root1234",
    "database": "attendance_system",
}

# Load trained models
with open("face_recognition_svm.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("known_face_encodings.pkl", "rb") as f:
    known_encodings = pickle.load(f)
with open("known_face_labels.pkl", "rb") as f:
    known_labels = pickle.load(f)

# Load anti-spoofing model
spoof_model = tf.keras.models.load_model("anti_spoofing_model.h5")

MODEL = "hog"
TOLERANCE = 0.35  

def is_real_face(frame):
    """ Predict if face is real or fake """
    frame_resized = cv2.resize(frame, (64, 64)) / 255.0  # Normalize
    frame_expanded = np.expand_dims(frame_resized, axis=0)
    prediction = spoof_model.predict(frame_expanded)
    
    return prediction[0][0] > 0.5  # Returns True if real, False if fake

def find_best_match(face_encoding):
    """ Find the best matching face using cosine similarity """
    similarities = cosine_similarity([face_encoding], known_encodings)
    best_match_idx = np.argmax(similarities)
    return known_labels[best_match_idx] if similarities[0][best_match_idx] > 0.6 else None

def check_attendance_status(roll_number):
    """ Check if attendance is already marked in the last hour """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = """
        SELECT timestamp FROM attendance 
        WHERE roll_number = %s 
        ORDER BY timestamp DESC LIMIT 1
        """
        cursor.execute(query, (roll_number,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result:
            last_attendance_time = result[0]
            current_time = datetime.now()
            time_difference = (current_time - last_attendance_time).total_seconds() / 3600  # Convert to hours
            return time_difference < 1  # True if already marked in the last hour
        return False
    except mysql.connector.Error as err:
        st.error(f"❌ Database Error: {err}")
        return False

def insert_attendance(roll_number, status):
    """ Insert attendance record into the database """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = "INSERT INTO attendance (roll_number, status, timestamp) VALUES (%s, %s, NOW())"
        cursor.execute(query, (roll_number, status))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as err:
        st.error(f"❌ Database Error: {err}")
        return False

st.title("📸 AI-Based Attendance System with Anti-Spoofing")

if st.button("Capture & Verify Attendance"):
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)  # Small delay to allow the camera to initialize
    
    if not cap.isOpened():
        st.error("Error: Could not access webcam.")
    else:
        ret, frame = cap.read()
        
        if ret:
            st.image(frame, channels="BGR", caption="Captured Image")

            # Convert BGR to RGB BEFORE detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face encodings
            face_encodings = face_recognition.face_encodings(rgb_frame)

            if face_encodings:
                recognized_roll = find_best_match(face_encodings[0])

                if recognized_roll:
                    # ✅ Check for spoofing attempt
                    if not is_real_face(frame):
                        st.error(f"❌ Spoofing detected! Marking {recognized_roll} as ABSENT.")
                        insert_attendance(recognized_roll, "ABSENT")  # Mark as absent for spoofing
                    else:
                        # ✅ Check if attendance was already marked in the last hour
                        if check_attendance_status(recognized_roll):
                            st.warning(f"⏳ Attendance already marked for {recognized_roll} in the last hour.")
                        else:
                            insert_attendance(recognized_roll, "PRESENT")
                            st.success(f"✅ Attendance marked for {recognized_roll}")
                else:
                    st.error("❌ No matching face found.")
            else:
                st.error("❌ No face detected in the image.")
        
        cap.release()
        cv2.destroyAllWindows()
