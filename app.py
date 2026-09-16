import streamlit as st
import os
import subprocess
import sys
import shutil
from datetime import datetime
import imageio_ffmpeg
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="AI CCTV Activity Monitor",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded"
)


BASE_DIR = r"D:\5th Sem CornerStoneProject"

VIDEO_DIR = os.path.join(
    BASE_DIR,
    "videos"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)
LIVE_OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(LIVE_OUTPUT_DIR, exist_ok=True)

BACKEND = os.path.join(
    BASE_DIR,
    "local_cctv.py"
)

os.makedirs(
    VIDEO_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# =========================================================
# SESSION STATE
# =========================================================
# ============================================================
# LIVE PC WEBCAM AI PROCESSOR
# ============================================================

ACTIVITY_CLASSES = ["Fall", "Run", "Sit", "Stand", "Walk"]

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_cnn_lstm_attention.pth"
)

YOLO_PATH = os.path.join(
    BASE_DIR,
    "models",
    "yolo11s.pt"
)


class CNNLSTMAttention(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()

        backbone = resnet18(weights=None)

        self.cnn = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            batch_first=True
        )

        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(
            256,
            num_classes
        )

    def forward(self, x):

        batch_size, seq_len = x.shape[:2]

        x = x.view(
            batch_size * seq_len,
            3,
            224,
            224
        )

        features = self.cnn(x)

        features = features.view(
            batch_size,
            seq_len,
            512
        )

        lstm_out, _ = self.lstm(features)

        attention_scores = self.attention(
            lstm_out
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        context = torch.sum(
            attention_weights * lstm_out,
            dim=1
        )

        context = self.dropout(context)

        return self.fc(context)


@st.cache_resource
def load_live_models():

    device = torch.device("cpu")

    activity_model = CNNLSTMAttention(
        num_classes=5
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        activity_model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=False
        )
    else:
        activity_model.load_state_dict(
            checkpoint,
            strict=False
        )

    activity_model.to(device)
    activity_model.eval()

    yolo_model = YOLO(YOLO_PATH)

    return activity_model, yolo_model


LIVE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class ActivityVideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.writer = None

        self.output_path = os.path.join(
            LIVE_OUTPUT_DIR,
            f"live_cctv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )

        self.model = None
        self.yolo = None

        self.buffers = defaultdict(
            lambda: deque(maxlen=16)
        )

        self.frame_count = 0
        self.last_output = None

        try:
            self.model, self.yolo = load_live_models()
            self.error = None

        except Exception as e:
            self.error = str(e)

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        # Initialize video writer
        if self.writer is None:

            height, width = img.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                15.0,
                (width, height)
            )

        self.frame_count += 1

        # Model loading error
        if self.error is not None:

            cv2.putText(
                img,
                "MODEL ERROR",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.putText(
                img,
                self.error[:100],
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

            if self.writer is not None:
                self.writer.write(img)

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )

        
        try:

            results = self.yolo.track(
                img,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=0.45,
                imgsz=320,
                verbose=False
            )

            if results and results[0].boxes is not None:

                boxes = results[0].boxes

                if boxes.id is not None:

                    ids = boxes.id.int().cpu().tolist()

                    xyxy = boxes.xyxy.int().cpu().tolist()

                    for track_id, box in zip(ids, xyxy):

                        x1, y1, x2, y2 = box

                        h, w = img.shape[:2]

                        pad_x = int(
                            (x2 - x1) * 0.30
                        )

                        pad_y = int(
                            (y2 - y1) * 0.30
                        )

                        x1c = max(
                            0,
                            x1 - pad_x
                        )

                        y1c = max(
                            0,
                            y1 - pad_y
                        )

                        x2c = min(
                            w,
                            x2 + pad_x
                        )

                        y2c = min(
                            h,
                            y2 + pad_y
                        )

                        crop = img[
                            y1c:y2c,
                            x1c:x2c
                        ]

                        if crop.size == 0:
                            continue

                        crop_rgb = cv2.cvtColor(
                            crop,
                            cv2.COLOR_BGR2RGB
                        )

                        tensor = LIVE_TRANSFORM(
                            crop_rgb
                        )

                        self.buffers[
                            track_id
                        ].append(tensor)

                        label = "Collecting..."

                        confidence = 0.0

                        # Activity prediction
                        if (
                            len(
                                self.buffers[track_id]
                            ) == 16
                            and self.frame_count % 8 == 0
                        ):

                            sequence = torch.stack(
                                list(
                                    self.buffers[track_id]
                                )
                            )

                            sequence = sequence.unsqueeze(0)

                            with torch.no_grad():

                                output = self.model(
                                    sequence
                                )

                                probabilities = torch.softmax(
                                    output,
                                    dim=1
                                )

                                confidence, prediction = torch.max(
                                    probabilities,
                                    dim=1
                                )

                            label = ACTIVITY_CLASSES[
                                prediction.item()
                            ]

                            confidence = (
                                confidence.item() * 100
                            )

                        # Box color
                        if label == "Fall":

                            box_color = (
                                0,
                                0,
                                255
                            )

                        else:

                            box_color = (
                                0,
                                200,
                                150
                            )

                        # Person bounding box
                        cv2.rectangle(
                            img,
                            (x1, y1),
                            (x2, y2),
                            box_color,
                            2
                        )

                        text = (
                            f"ID {track_id:02d} | "
                            f"{label} | "
                            f"{confidence:.0f}%"
                        )

                        # Label background
                        label_y1 = max(
                            0,
                            y1 - 32
                        )

                        label_y2 = y1

                        label_x2 = min(
                            w,
                            x1 + 280
                        )

                        cv2.rectangle(
                            img,
                            (x1, label_y1),
                            (label_x2, label_y2),
                            box_color,
                            -1
                        )

                        # Label text
                        cv2.putText(
                            img,
                            text,
                            (
                                x1 + 5,
                                max(22, y1 - 10)
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2
                        )

                        # Fall alert
                        if label == "Fall":

                            cv2.putText(
                                img,
                                "FALL ALERT",
                                (
                                    x1,
                                    min(
                                        h - 10,
                                        y2 + 30
                                    )
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2
                            )

            # Monitoring indicator
            cv2.putText(
                img,
                "LIVE AI MONITORING",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 150),
                2
            )

        except Exception as e:

            cv2.putText(
                img,
                f"Processing error: {str(e)[:70]}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        # Save processed frame
        if self.writer is not None:

            self.writer.write(img)

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

    def __del__(self):

        if self.writer is not None:

            self.writer.release()

    
if "output_video" not in st.session_state:

    st.session_state.output_video = None


if "processing" not in st.session_state:

    st.session_state.processing = False


if "saved" not in st.session_state:

    st.session_state.saved = False


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #E5E7EB;
}
/* System Status - readable on light background */

[data-testid="stMetricLabel"] {
    color: #374151 !important;
}

[data-testid="stMetricLabel"] p {
    color: #374151 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-size: 30px !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] div {
    color: #111827 !important;
}
.stSubheader {
    color: #111827 !important;
}
h3 {
    color: #111827 !important;
}

.header {
    background: #1F2937;
    padding: 28px;
    text-align: center;
    border-radius: 8px;
    margin-bottom: 25px;
}

.header h1 {
    color: #FFFFFF;
    margin: 0;
    font-size: 38px;
    font-weight: 700;
}

.header p {
    color: #E5E7EB;
    margin: 7px 0;
}

.student-panel {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-left: 5px solid #166534;
    border-radius: 8px;
    padding: 22px 24px;
    margin-bottom: 16px;
}

.student-panel-title {
    color: #111827;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 18px;
}

.student-label {
    color: #111827;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .5px;
    margin-bottom: 6px;
}

.student-value {
    color: #111827;
    font-size: 17px;
    font-weight: 600;
    line-height: 1.4;
}

.project-meta {
    color: #6B7280;
    font-size: 15px;
    margin-top: 7px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #111827;
    border-bottom: 2px solid #D1D5DB;
    padding-bottom: 9px;
    margin-top: 16px;
    margin-bottom: 10px;
}

.info-card {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 8px;
    padding: 18px;
    margin-bottom: 10px;
}.info-card b {
    color: #111827 !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}
.alert-card {
    background-color: #FEE2E2;
    border: 1px solid #DC2626;
    border-radius: 8px;
    padding: 18px;
    color: #991B1B;
}

.stButton > button {
    background-color: #166534;
    color: #FFFFFF;
    border: 1px solid #14532D;
    font-weight: 600;
    min-height: 44px;
}

.stButton > button:hover {
    background-color: #15803D;
    color: #FFFFFF;
}

.stDownloadButton > button {
    background-color: #1F2937;
    color: #FFFFFF;
    font-weight: 600;
}


/* Improve readability of Streamlit native controls on the light theme */
.stRadio label,
.stSelectbox label,
.stTextInput label,
.stFileUploader label,
.stCheckbox label,
.stToggle label {
    color: #374151 !important;
}

.stRadio [data-testid="stWidgetLabel"] p,
.stSelectbox [data-testid="stWidgetLabel"] p,
.stTextInput [data-testid="stWidgetLabel"] p,
.stFileUploader [data-testid="stWidgetLabel"] p,
.stCheckbox [data-testid="stWidgetLabel"] p,
.stToggle [data-testid="stWidgetLabel"] p {
    color: #111827 !important;
    font-weight: 600 !important;
}

.stFileUploader section {
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
}
.stFileUploader section button {
    background-color: #166534 !important;
    color: #FFFFFF !important;
    border: 1px solid #14532D !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    min-height: 40px !important;
}

.stFileUploader section * {
    color: #FFFFFF !important;
}

.stFileUploader section button {
    background-color: #1F2937 !important;
    color: #FFFFFF !important;
    border: 1px solid #374151 !important;
}

.stFileUploader section small {
    color: #6B7280 !important;
}

.stAlert {
    border-radius: 8px !important;
}

[data-testid="stAlert"] p {
    color: #374151 !important;
}

[data-testid="stAlert"][kind="success"] p {
    color: #166534 !important;
}

[data-testid="stAlert"][kind="error"] p {
    color: #991B1B !important;
}

[data-testid="stAlert"][kind="warning"] p {
    color: #92400E !important;
}

[data-testid="stAlert"][kind="info"] p {
    color: #1D4ED8 !important;
}

.stSubheader {
    color: #111827 !important;
}

div[data-baseweb="radio"] label {
    color: #374151 !important;
}

div[data-baseweb="radio"] label p {
    color: #374151 !important;
}

div[data-baseweb="select"] * {
    color: #374151 !important;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
<div class="header">

<h1>AI BASED CCTV ACTIVITY MONITOR</h1>

<p style="font-size:18px; font-weight:500;">
Intelligent Vision-Based Human Activity Recognition
</p>

<p style="font-size:17px;">
SYSTEM
</p>

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# STUDENT / PROJECT INFORMATION
# =========================================================
st.markdown(
    """
    <div class="student-panel">
        <div class="student-panel-title">Student & Project Information</div>
        <div style="display:grid; grid-template-columns:1.15fr 2fr 1.25fr 1fr; gap:24px;">
            <div>
                <div class="student-label">Student Name</div>
                <div class="student-value">Brijendra Vishwakarma</div>
            </div>
            <div>
                <div class="student-label">College</div>
                <div class="student-value">Madhav Institute of Technology and Science, Gwalior</div>
            </div>
            <div>
                <div class="student-label">Enrollment No.</div>
                <div class="student-value">BTIR24O1022</div>
            </div>
            <div>
                <div class="student-label">Branch / Batch</div>
                <div class="student-value">AIR — 2028</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# INPUT SOURCE
# =========================================================

st.markdown(
    '<div class="section-title">Input Source</div>',
    unsafe_allow_html=True
)


source = st.radio(
    "Monitoring mode",
    [
        "Video Upload",
        "Live Camera"
    ],
    horizontal=True
)


# =========================================================
# VIDEO UPLOAD
# =========================================================

uploaded_file = None
input_path = None


if source == "Video Upload":

    st.markdown(
        '<div class="info-card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "CCTV Video"
    )

    uploaded_file = st.file_uploader(
        "Upload CCTV footage",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )

    if uploaded_file:

        input_path = os.path.join(
            VIDEO_DIR,
            uploaded_file.name
        )

        with open(
            input_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        st.success(
            "Video uploaded successfully."
        )

        st.video(
            input_path
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# LIVE CAMERA
# =========================================================

else:

    st.markdown(
        '<div class="info-card">',
        unsafe_allow_html=True
    )

    st.subheader("Live Camera")

    st.info(
        "Use your PC webcam for real-time AI activity monitoring. "
        "Allow camera access when your browser asks."
    )

    try:

        webrtc_streamer(
            key="pc-webcam-ai-monitor",
            video_processor_factory=ActivityVideoProcessor,
            media_stream_constraints={
                 "video": {
        "width": {"ideal": 640},
        "height": {"ideal": 480},
        "frameRate": {"ideal": 15}
    },
    "audio": False
            },
            async_processing=True
        )

    except Exception as camera_error:

        st.error(
            f"Webcam initialization failed: {camera_error}"
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# RECORDING
# =========================================================

st.markdown(
    '<div class="section-title">Recording</div>',
    unsafe_allow_html=True
)


save_recording = st.toggle(
    "Save Recording",
    value=True
)


if save_recording:

    st.success(
        "Recording is ON — annotated video will be saved."
    )

else:

    st.info(
        "Recording is OFF — output will be displayed "
        "only after processing."
    )


# =========================================================
# CONTROLS
# =========================================================

st.markdown(
    '<div class="section-title">Monitoring Controls</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:

    start_detection = st.button(
        "Start Detection",
        use_container_width=True,
        type="primary"
    )


with col2:

    clear_result = st.button(
        "Clear Result",
        use_container_width=True
    )


if clear_result:

    st.session_state.output_video = None

    st.session_state.saved = False

    st.rerun()


# =========================================================
# DETECTION
# =========================================================
def convert_to_browser_mp4(input_path, output_path):

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        output_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return output_path

if start_detection:

    if source == "Live Camera":
        st.info("Live Camera mode is available below. Use the webcam section to start live monitoring.")

    elif uploaded_file is None:
        st.warning("Please upload a CCTV video first.")

    else:
        st.session_state.processing = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Raw output generated by OpenCV
        raw_output_path = os.path.join(
            RESULTS_DIR,
            f"cctv_result_{timestamp}.mp4"
        )

        # Browser-compatible output
        browser_output_path = os.path.join(
            RESULTS_DIR,
            f"browser_cctv_result_{timestamp}.mp4"
        )

        environment = os.environ.copy()

        environment["CCTV_INPUT_VIDEO"] = input_path
        environment["CCTV_OUTPUT_VIDEO"] = raw_output_path
        environment["CCTV_SAVE_RECORDING"] = (
            "1" if save_recording else "0"
        )

        st.info("AI detection is running. Please wait...")

        try:

            process = subprocess.Popen(
                [sys.executable, BACKEND],
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment
            )

            progress_placeholder = st.empty()


            for line in process.stdout:

                line = line.strip()

                if line:

                    # Detect progress information
                    if "Processing" in line:
                        progress_placeholder.info(line)

            process.wait()

            # --------------------------------------------------
            # SUCCESS
            # --------------------------------------------------

            if process.returncode == 0:

                st.session_state.processing = False

                if save_recording and os.path.exists(raw_output_path):

                    try:

                        # Convert OpenCV MP4 into browser-compatible H.264 MP4
                        convert_to_browser_mp4(
                            raw_output_path,
                            browser_output_path
                        )

                        # Use browser-compatible video for Streamlit
                        if os.path.exists(browser_output_path):

                            st.session_state.output_video = (
                                browser_output_path
                            )

                        else:

                            st.session_state.output_video = (
                                raw_output_path
                            )

                        st.session_state.saved = True

                        st.success(
                            "Detection completed and recording saved."
                        )

                    except Exception as conversion_error:

                        # If conversion fails, still keep original result
                        st.session_state.output_video = raw_output_path
                        st.session_state.saved = True

                        st.warning(
                            "Detection completed, but browser conversion "
                            "failed. The original recording was saved."
                        )

                        st.caption(
                            f"Conversion error: {conversion_error}"
                        )

                else:

                    st.session_state.output_video = None
                    st.session_state.saved = False

                    st.success(
                        "Detection completed."
                    )

                    if not save_recording:

                        st.info(
                            "Save Recording was OFF, so no output "
                            "video was stored."
                        )

            # --------------------------------------------------
            # FAILURE
            # --------------------------------------------------

            else:

                st.session_state.processing = False
                st.session_state.output_video = None
                st.session_state.saved = False

                st.error(
                    "AI detection failed."
                )

                if output_lines:

                    st.code(
                        "\n".join(output_lines[-20:])
                    )

        except Exception as error:

            st.session_state.processing = False
            st.session_state.output_video = None
            st.session_state.saved = False

            st.error(
                f"An error occurred while running AI detection:\n\n{error}"
            )


# =========================================================
# RESULT
# =========================================================

st.markdown(
    '<div class="section-title">Detection Result</div>',
    unsafe_allow_html=True
)


if (
    st.session_state.output_video
    and os.path.exists(
        st.session_state.output_video
    )
):

    st.video(
        st.session_state.output_video
    )

    st.success(
        "Annotated CCTV recording is ready."
    )

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        if st.button(
            "Show Recording",
            use_container_width=True
        ):

            st.video(
                st.session_state.output_video
            )


    with result_col2:

        with open(
            st.session_state.output_video,
            "rb"
        ) as video_file:

            st.download_button(
                "Download Recording",
                video_file,
                file_name=os.path.basename(
                    st.session_state.output_video
                ),
                mime="video/mp4",
                use_container_width=True
            )


else:

    st.info(
        "No processed recording available."
    )


# =========================================================
# SYSTEM STATUS
# =========================================================



def convert_to_browser_mp4(input_path, output_path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return output_path
# =========================================================
# ACTIVITIES
# =========================================================

st.markdown(
    '<div class="section-title">Supported Activities</div>',
    unsafe_allow_html=True
)


activities = [
    "FALL",
    "RUN",
    "SIT",
    "STAND",
    "WALK"
]


cols = st.columns(5)

for col, activity in zip(
    cols,
    activities
):

    with col:

        st.markdown(
            f"""
            <div class="info-card"
                 style="text-align:center;">
            <b>{activity}</b>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# ALERT MONITOR
# =========================================================

st.markdown(
    '<div class="section-title">Alert Monitor</div>',
    unsafe_allow_html=True
)


st.info(
    "Fall alerts are generated when the "
    "activity model predicts FALL for a tracked person."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="header" style="margin-top:35px;">

<b>B.Tech — Artificial Intelligence & Robotics</b>

<br><br>

Design and Development of an Intelligent
Vision-Based Human Activity Recognition
Framework Using Deep Learning

<br><br>

Madhav Institute of Technology and Science

</div>
""",
    unsafe_allow_html=True
)