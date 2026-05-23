# SoundGuard AI — Streamlit MVP Real-Time Audio App

An automated real-time sound monitoring dashboard built on Streamlit using Google's YAMNet neural network model to track audio anomalies.

## Project Workspace Structure
```text
your_project/
│
├── app.py                  # Streamlit application core script
├── sound_events.db         # Self-initializing SQLite history tracking ledger
├── requirements.txt        # Python dependency manifest
└── yamnet_model/           # Unpacked directory containing YAMNet weights
    ├── saved_model.pb
    ├── variables/
    └── assets/
        └── yamnet_class_map.csv
```

## System Deployment & Launch Setup

1. **System Dependencies Setup (Linux/Ubuntu configurations):**
When deploying onto Linux-driven architecture, `sounddevice` runtime contexts require the installation of underlying system bindings (`portaudio` library):
```bash
sudo apt-get update && sudo apt-get install portaudio19-dev
```

2. **Install Package Manifest Environments:**
Execute a package sync within your virtual ecosystem environment:
```bash
pip install -r requirements.txt
```

3. **Boot Up Application Terminal:**
Fire up the local host web server processing node using the command below:
```bash
streamlit run app.py
```

## Functional Scope Blueprint

* **Universal Class Access:** Pulls all 500+ standard audio categories dynamically out of the YAMNet map configuration array for selective multi-choice targeting on the fly.
* **Hardware Integration Node:** Probes standard system sound servers to switch stream pipelines between active input hardware interfaces.
* **Asynchronous Web Execution Threading:** Offloads heavy feature processing array metrics onto background workers to render metrics and line graphs smoothly in the interface.