# Intelligent Human Activity Recognition

## Design and Development of an Intelligent Vision-Based Human Activity Recognition Framework Using Deep Learning

A vision-based CCTV activity recognition system that detects and tracks people and recognizes their activities using deep learning.

The system combines **YOLO11s**, **ByteTrack**, and a **CNN-LSTM model with temporal attention** to analyze human activities from CCTV video and webcam streams.

---

## Features

- Person detection using YOLO11s
- Multi-person tracking using ByteTrack
- Anonymous person IDs such as `ID 01`, `ID 02`
- Human activity recognition
- CNN-LSTM based temporal modeling
- Temporal attention mechanism
- Activity confidence scores
- Fall activity alert
- Annotated CCTV output video
- Video upload support
- PC webcam support
- Streamlit-based monitoring dashboard
- Activity monitoring for multiple people

---

## Supported Activities

The current model recognizes five activities:

- Fall
- Run
- Sit
- Stand
- Walk

---

## System Architecture

```text
CCTV / Webcam
      |
      v
YOLO11s Person Detection
      |
      v
ByteTrack Multi-Object Tracking
      |
      v
Person-Specific Frame Sequence
      |
      v
ResNet18 Feature Extraction
      |
      v
LSTM Temporal Modeling
      |
      v
Temporal Attention
      |
      v
Activity Classification
      |
      +----------------+
      |                |
      v                v
Activity Label      Alert System
      |
      v
Streamlit Dashboard
```

## Technologies Used
Python
PyTorch
Torchvision
OpenCV
Ultralytics YOLO
ByteTrack
Streamlit
Streamlit-WebRTC
NumPy
## Deep Learning Model

The activity recognition model uses the following architecture:
```
ResNet18
   |
   v
Feature Extraction
   |
   v
LSTM
   |
   v
Temporal Attention
   |
   v
Fully Connected Layer
   |
   v
5 Activity Classes
```

The ResNet18 backbone extracts visual features from video frames. The LSTM models temporal information across a sequence of frames, while temporal attention helps the model focus on important frames.

## Model Performance

The current CNN-LSTM-Attention model achieved:

Validation Accuracy: 55.05%

The current validation confusion matrix shows that activities such as Stand and Walk, as well as Fall and Run, can sometimes be confused.

Further dataset improvement and training are planned to improve recognition performance.

## Project Structure
```
Intelligent-Human-Activity-Recognition/
│
├── app.py
├── local_cctv.py
├── test_model.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── best_cnn_lstm_attention.pth
│   └── yolo11s.pt
│
├── videos/
│   └── test videos
│
└── results/
    └── generated output videos
```

Large model files, datasets, videos, and generated results are not included in this GitHub repository.

## Installation
1. Clone the repository
```git clone https://github.com/brijendr avishwakarma/Intelligent-Human-Activity-Recognition.git ```
2. Open the project directory
```cd Intelligent-Human-Activity-Recognition```
3. Create a virtual environment
```python -m venv venv```
4. Activate the virtual environment

Windows PowerShell:

```.\venv\Scripts\Activate.ps1```
5. Install dependencies
```pip install -r requirements.txt```
## Model Setup

The trained model files are required to run the complete system.

Place the following files inside the models directory:

```models/
├── best_cnn_lstm_attention.pth
└── yolo11s.pt
```

The large model files are intentionally excluded from the Git repository.

## Running the Application

Start the Streamlit application:

```streamlit run app.py```

The application provides:

Video Upload
Live Camera
CCTV activity detection
Person tracking
Activity recognition
Fall alerts
Annotated output recording

## CCTV Processing Pipeline

For uploaded CCTV videos, the system performs:

Read the CCTV video.
Detect people using YOLO11s.
Track detected people using ByteTrack.
Maintain frame sequences for individual tracked persons.
Extract visual features.
Analyze temporal information using LSTM.
Apply temporal attention.
Predict the current activity.
Display activity labels above detected persons.
Generate alerts for fall activity.
Save the annotated CCTV recording.
## Alert System

The system currently uses Fall as the primary alert activity.

Example:

```ID 03
FALL
Confidence: 82%
```

The alert mechanism is designed to help identify potentially dangerous events in CCTV footage.

## Privacy

The system uses anonymous tracking IDs such as:

```ID 01
ID 02
ID 03
```

It does not require face recognition or personal identity recognition for activity tracking.

## Future Improvements

Possible future improvements include:

Improved activity recognition accuracy
Larger and more diverse training dataset
Better real-time CPU performance
Activity history and event logging
Real-time statistics
Improved alert management
Additional activities such as fighting
Email/SMS notification support
Deployment on edge devices
Improved temporal modeling architectures
