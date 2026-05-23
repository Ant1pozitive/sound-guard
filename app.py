import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import sqlite3
import datetime
import os
import sys
import csv
import scipy.signal
import threading
import time
import streamlit as st
import pandas as pd

# --- INITIALIZE STREAMLIT PAGE ---
st.set_page_config(page_title="SoundGuard AI MVP", layout="wide")
st.title("🔊 SoundGuard AI — Real-Time Audio Anomaly Monitoring")

MODEL_DIR = "yamnet_model"
DB_NAME = "sound_events.db"
TARGET_SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5
WINDOW_DURATION = 0.96
WINDOW_SAMPLES_TARGET = int(TARGET_SAMPLE_RATE * WINDOW_DURATION)

# System validation for the model folder
if not os.path.exists(MODEL_DIR):
    st.error(f"Model directory '{MODEL_DIR}' not found. Please download it from Kaggle/TFHub and place it here.")
    st.stop()

# --- CACHE MODEL LOADING ---
@st.cache_resource
def load_yamnet():
    model = hub.load(MODEL_DIR)
    class_map_path = os.path.join(MODEL_DIR, "assets", "yamnet_class_map.csv")
    if not os.path.exists(class_map_path):
        st.error(f"Class map file missing at: '{class_map_path}'")
        st.stop()
        
    with open(class_map_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        # Load absolutely ALL classes from the YAMNet class map
        classes = [row[2] for row in reader]
    return model, classes

model, all_class_names = load_yamnet()

# --- INITIALIZE DATABASE ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- AUDIO SYSTEM UTILITIES ---
def find_input_devices():
    devices = sd.query_devices()
    input_devices = {}
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices[f"{i}: {dev['name']}"] = i
    return input_devices

def get_target_indices(selected_classes):
    """Maps selected class names to their respective YAMNet indices"""
    target_indices = {}
    for class_name in selected_classes:
        indices = [i for i, name in enumerate(all_class_names) if name == class_name]
        if indices:
            target_indices[class_name] = indices
    return target_indices

# --- BACKGROUND WORKER FOR AUDIO CAPTURE & INFERENCE ---
class AudioBackgroundWorker:
    def __init__(self):
        self.is_running = False
        self.thread = None
        self.audio_buffer = np.zeros(0, dtype=np.float32)

    def start(self, device_id, target_indices, threshold):
        if not self.is_running:
            self.is_running = True
            self.target_indices = target_indices
            self.threshold = threshold
            self.device_id = device_id
            self.audio_buffer = np.zeros(0, dtype=np.float32)
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _run(self):
        try:
            dev_info = sd.query_devices(self.device_id)
            device_samplerate = int(dev_info['default_samplerate'])
        except Exception as e:
            print(f"Audio device initialization error: {e}")
            self.is_running = False
            return

        chunk_samples_device = int(device_samplerate * CHUNK_DURATION)
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        try:
            with sd.InputStream(device=self.device_id, samplerate=device_samplerate,
                                 channels=1, dtype='float32', blocksize=chunk_samples_device) as stream:
                while self.is_running:
                    chunk, overflowed = stream.read(chunk_samples_device)
                    
                    if chunk.ndim > 1:
                        chunk = np.mean(chunk, axis=1)
                    else:
                        chunk = chunk.flatten()

                    target_len = int(len(chunk) * TARGET_SAMPLE_RATE / device_samplerate)
                    chunk_resampled = scipy.signal.resample(chunk, target_len)
                    self.audio_buffer = np.concatenate([self.audio_buffer, chunk_resampled])

                    if len(self.audio_buffer) < WINDOW_SAMPLES_TARGET:
                        continue

                    window = self.audio_buffer[-WINDOW_SAMPLES_TARGET:]
                    self.audio_buffer = self.audio_buffer[-WINDOW_SAMPLES_TARGET:]

                    # YAMNet Inference
                    scores, _, _ = model(window)
                    frame_scores = scores.numpy()[0]

                    detections = {}
                    for anomaly_name, indices in self.target_indices.items():
                        max_conf = np.max(frame_scores[indices])
                        if max_conf >= self.threshold:
                            detections[anomaly_name] = float(max_conf)

                    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    if detections:
                        for event, conf in detections.items():
                            cursor.execute(
                                "INSERT INTO events (timestamp, event_type, confidence) VALUES (?, ?, ?)",
                                (timestamp_str, event, conf)
                            )
                        conn.commit()
                    else:
                        # Log normal state to keep chart timelines consistent
                        cursor.execute(
                            "INSERT INTO events (timestamp, event_type, confidence) VALUES (?, ?, ?)",
                            (timestamp_str, 'Normal', 0.0)
                        )
                        conn.commit()
                        
        except Exception as e:
            print(f"Exception in audio thread: {e}")
        finally:
            conn.close()

# Keep the background thread persistent across Streamlit script reruns
if 'worker' not in st.session_state:
    st.session_state.worker = AudioBackgroundWorker()

# --- UI SIDEBAR LAYOUT ---
sidebar = st.sidebar
sidebar.header("🎛 Monitoring Configuration")

available_devices = find_input_devices()
if not available_devices:
    st.error("No active audio input devices found.")
    st.stop()

selected_device_label = sidebar.selectbox("Audio Input Device", list(available_devices.keys()))
device_id = available_devices[selected_device_label]

# Multi-select dropdown featuring all 521 YAMNet classes
selected_classes = sidebar.multiselect(
    "Tracked Audio Classes",
    options=all_class_names,
    default=["Gunshot", "Scream", "Glass", "Music"] if all(c in all_class_names for c in ["Gunshot", "Scream", "Glass", "Music"]) else [all_class_names[0]]
)

threshold = sidebar.slider("Detection Threshold (Confidence)", 0.0, 1.0, 0.3, 0.05)
refresh_rate = sidebar.slider("UI Refresh Interval (Seconds)", 0.2, 2.0, 0.5, 0.1)

# Stream lifecycle control buttons
if st.session_state.worker.is_running:
    if sidebar.button("🔴 STOP MONITORING", use_container_width=True):
        st.session_state.worker.stop()
        st.rerun()
else:
    if sidebar.button("🟢 START MONITORING", use_container_width=True):
        target_indices = get_target_indices(selected_classes)
        if not target_indices:
            st.sidebar.error("Please select at least one audio class to monitor!")
        else:
            st.session_state.worker.start(device_id, target_indices, threshold)
            st.rerun()

# --- MAIN REAL-TIME MONITORING UI ---
status_col, indicator_col = st.columns([3, 1])
with status_col:
    if st.session_state.worker.is_running:
        st.success(f"System Status: ACTIVE | Streaming via: {selected_device_label}")
    else:
        st.warning("System Status: STOPPED. Use the sidebar controls to activate the stream.")

# Placeholders for dynamic rendering
chart_placeholder = st.empty()
table_header = st.empty()
table_placeholder = st.empty()

# --- DASHBOARD AUTO-REFRESH RECYCLE LOOP ---
while st.session_state.worker.is_running:
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(
            "SELECT timestamp, event_type, confidence FROM events ORDER BY id DESC LIMIT 40", 
            conn
        )
        conn.close()

        if not df.empty:
            df_chart = df.iloc[::-1].copy()

            # 1. Render Real-time Stream Charts
            with chart_placeholder.container():
                st.subheader("📊 Metrics Streaming (Recent Window)")
                chart_data = df_chart.pivot_table(
                    index='timestamp', 
                    columns='event_type', 
                    values='confidence', 
                    aggfunc='max'
                ).fillna(0.0)
                
                if 'Normal' in chart_data.columns and len(chart_data.columns) > 1:
                    chart_data = chart_data.drop(columns=['Normal'])
                st.line_chart(chart_data, height=300)

            # 2. Render Incident Tables
            df_anomalies = df[df['event_type'] != 'Normal']
            
            with table_header.container():
                st.subheader(f"🚨 Critical Logs Registry (Detections: {len(df_anomalies)})")

            if not df_anomalies.empty:
                table_placeholder.dataframe(
                    df_anomalies.style.format({'confidence': '{:.2f}'})
                    .background_gradient(cmap='Reds', subset=['confidence']),
                    use_container_width=True
                )
            else:
                table_placeholder.info("All channels clear. No active anomalies matching criteria recorded in this window.")
                
    except Exception as e:
        print(f"UI refresh cycle skip: {e}")

    time.sleep(refresh_rate)