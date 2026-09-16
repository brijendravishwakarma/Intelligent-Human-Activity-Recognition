import os
import cv2
import torch
import numpy as np
import torch.nn as nn

from collections import defaultdict, deque
from ultralytics import YOLO
from torchvision.models import resnet18


# =========================================================
# PATHS
# =========================================================

BASE_DIR = r"D:\5th Sem CornerStoneProject"

YOLO_PATH = os.path.join(
    BASE_DIR,
    "models",
    "yolo11s.pt"
)

ACTIVITY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_cnn_lstm_attention.pth"
)

DEFAULT_INPUT = os.path.join(
    BASE_DIR,
    "videos",
    "NTU_fight0003_fall_1.mp4"
)

DEFAULT_OUTPUT = os.path.join(
    BASE_DIR,
    "results",
    "local_cctv_demo.mp4"
)

INPUT_VIDEO = os.environ.get(
    "CCTV_INPUT_VIDEO",
    DEFAULT_INPUT
)

OUTPUT_VIDEO = os.environ.get(
    "CCTV_OUTPUT_VIDEO",
    DEFAULT_OUTPUT
)

SAVE_RECORDING = os.environ.get(
    "CCTV_SAVE_RECORDING",
    "1"
) == "1"


os.makedirs(
    os.path.dirname(OUTPUT_VIDEO),
    exist_ok=True
)


# =========================================================
# SETTINGS
# =========================================================

DEVICE = torch.device("cpu")

NUM_FRAMES = 16
PREDICT_EVERY = 8
PADDING = 0.30
IMAGE_SIZE = 224

CLASSES = [
    "fall",
    "run",
    "sit",
    "stand",
    "walk"
]


# =========================================================
# MODEL
# =========================================================

class CNNLSTMAttention(nn.Module):

    def __init__(
        self,
        num_classes=5,
        hidden_size=256
    ):

        super().__init__()

        self.cnn = resnet18(
            weights=None
        )

        self.cnn.fc = nn.Identity()

        for param in self.cnn.layer1.parameters():
            param.requires_grad = False

        for param in self.cnn.layer2.parameters():
            param.requires_grad = False

        for param in self.cnn.layer3.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        batch_size, frames, C, H, W = x.shape

        x = x.view(
            batch_size * frames,
            C,
            H,
            W
        )

        features = self.cnn(x)

        features = features.view(
            batch_size,
            frames,
            -1
        )

        lstm_out, _ = self.lstm(
            features
        )

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

        context = self.dropout(
            context
        )

        return self.fc(context)


# =========================================================
# LOAD MODELS
# =========================================================

print("Loading activity model...")

activity_model = CNNLSTMAttention(
    num_classes=5,
    hidden_size=256
)

checkpoint = torch.load(
    ACTIVITY_MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    activity_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    activity_model.load_state_dict(
        checkpoint
    )

activity_model.to(DEVICE)
activity_model.eval()

print("Activity model loaded.")


print("Loading YOLO11s...")

yolo = YOLO(
    YOLO_PATH
)

print("YOLO11s loaded.")


# =========================================================
# CROP FUNCTION
# =========================================================

def crop_person(frame, box):

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = map(
        int,
        box
    )

    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = int(
        box_width * PADDING
    )

    pad_y = int(
        box_height * PADDING
    )

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        width,
        x2 + pad_x
    )

    y2 = min(
        height,
        y2 + pad_y
    )

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    crop = cv2.resize(
        crop,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    return crop


# =========================================================
# PREPARE SEQUENCE
# =========================================================

def prepare_sequence(frames):

    arr = np.stack(
        frames
    ).astype(
        np.float32
    ) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    arr = (
        arr - mean
    ) / std

    arr = np.transpose(
        arr,
        (0, 3, 1, 2)
    )

    tensor = torch.from_numpy(
        arr.copy()
    )

    return tensor.unsqueeze(0)


# =========================================================
# OPEN VIDEO
# =========================================================

print()
print("=" * 60)
print("STARTING CCTV ACTIVITY RECOGNITION")
print("=" * 60)

print("Input:", INPUT_VIDEO)

cap = cv2.VideoCapture(
    INPUT_VIDEO
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video:\n{INPUT_VIDEO}"
    )


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30.0

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

print(
    f"Resolution: {width}x{height}"
)

print(
    f"FPS: {fps}"
)

print(
    f"Frames: {total_frames}"
)

print(
    "Save recording:",
    SAVE_RECORDING
)


# =========================================================
# VIDEO WRITER
# =========================================================

writer = None

if SAVE_RECORDING:

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )


# =========================================================
# BUFFERS
# =========================================================

person_buffers = defaultdict(
    lambda: deque(
        maxlen=NUM_FRAMES
    )
)

person_predictions = {}

frame_number = 0

alert_frames = 0

max_persons = 0

all_ids = set()


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = yolo.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.40,
        verbose=False,
        device="cpu"
    )

    result = results[0]

    current_ids = []

    fall_detected = False

    fall_ids = []

    if result.boxes.id is not None:

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        current_ids = ids.tolist()

        all_ids.update(
            current_ids
        )

        max_persons = max(
            max_persons,
            len(current_ids)
        )

        for box, track_id in zip(
            boxes,
            ids
        ):

            crop = crop_person(
                frame,
                box
            )

            if crop is None:
                continue

            person_buffers[
                track_id
            ].append(crop)

            # ---------------------------------------------
            # PREDICT
            # ---------------------------------------------

            if (
                len(
                    person_buffers[track_id]
                ) == NUM_FRAMES
                and frame_number % PREDICT_EVERY == 0
            ):

                sequence = prepare_sequence(
                    person_buffers[track_id]
                )

                with torch.no_grad():

                    output = activity_model(
                        sequence
                    )

                    probabilities = torch.softmax(
                        output,
                        dim=1
                    )

                    confidence, prediction = (
                        probabilities.max(
                            dim=1
                        )
                    )

                activity = CLASSES[
                    prediction.item()
                ]

                confidence_value = (
                    confidence.item() * 100
                )

                person_predictions[
                    track_id
                ] = (
                    activity,
                    confidence_value
                )

            # ---------------------------------------------
            # DRAW BOX
            # ---------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box
            )

            activity = None
            confidence = 0

            if track_id in person_predictions:

                activity, confidence = (
                    person_predictions[
                        track_id
                    ]
                )

                if activity == "fall":

                    fall_detected = True

                    fall_ids.append(
                        track_id
                    )

            # ---------------------------------------------
            # BOX COLOR
            # ---------------------------------------------

            if activity == "fall":

                box_color = (
                    0,
                    0,
                    255
                )

            elif activity == "run":

                box_color = (
                    0,
                    165,
                    255
                )

            elif activity == "walk":

                box_color = (
                    255,
                    200,
                    0
                )

            elif activity == "sit":

                box_color = (
                    255,
                    100,
                    100
                )

            elif activity == "stand":

                box_color = (
                    0,
                    200,
                    100
                )

            else:

                box_color = (
                    255,
                    255,
                    255
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                3 if activity == "fall" else 2
            )

            # ---------------------------------------------
            # LABEL
            # ---------------------------------------------

            if activity is not None:

                label = (
                    f"ID {track_id} | "
                    f"{activity.upper()} | "
                    f"{confidence:.0f}%"
                )

            else:

                label = (
                    f"ID {track_id} | ANALYZING"
                )

            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = 0.60

            thickness = 2

            (tw, th), _ = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            label_y = max(
                y1 - 10,
                th + 10
            )

            cv2.rectangle(
                frame,
                (
                    x1,
                    label_y - th - 8
                ),
                (
                    x1 + tw + 10,
                    label_y + 5
                ),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                label,
                (
                    x1 + 5,
                    label_y
                ),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

    # =====================================================
    # HEADER
    # =====================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 70),
        (15, 20, 25),
        -1
    )

    cv2.putText(
        frame,
        "AI CCTV ACTIVITY MONITOR",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Persons: {len(current_ids)}",
        (20, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (width - 220, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    # =====================================================
    # ALERT
    # =====================================================

    if fall_detected:

        alert_frames += 1

        alert_text = (
            "WARNING: FALL DETECTED"
        )

        if fall_ids:

            alert_text += (
                " | ID "
                + ", ".join(
                    map(
                        str,
                        fall_ids
                    )
                )
            )

        cv2.rectangle(
            frame,
            (0, height - 55),
            (width, height),
            (0, 0, 180),
            -1
        )

        cv2.putText(
            frame,
            alert_text,
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    else:

        cv2.rectangle(
            frame,
            (0, height - 40),
            (width, height),
            (15, 20, 25),
            -1
        )

        cv2.putText(
            frame,
            "System monitoring active",
            (20, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

    # =====================================================
    # WRITE OUTPUT
    # =====================================================

    if writer is not None:

        writer.write(
            frame
        )

    # =====================================================
    # PROGRESS
    # =====================================================

    print(
        f"\rProcessing "
        f"{frame_number}/{total_frames}",
        end=""
    )


# =========================================================
# CLEANUP
# =========================================================

cap.release()

if writer is not None:

    writer.release()


print()
print()
print("=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)

print(
    "Frames processed:",
    frame_number
)

print(
    "Unique persons:",
    len(all_ids)
)

print(
    "Maximum persons in a frame:",
    max_persons
)

print(
    "Alert frames:",
    alert_frames
)

if SAVE_RECORDING:

    print(
        "Saved recording:",
        OUTPUT_VIDEO
    )

else:

    print(
        "Recording was not saved."
    )