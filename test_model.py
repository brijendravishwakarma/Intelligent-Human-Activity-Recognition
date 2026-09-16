import torch
import torch.nn as nn
from torchvision.models import resnet18


class CNNLSTMAttention(nn.Module):

    def __init__(self, num_classes=5, hidden_size=256):

        super().__init__()

        self.cnn = resnet18(weights=None)

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

        batch_size, num_frames, C, H, W = x.shape

        x = x.view(
            batch_size * num_frames,
            C,
            H,
            W
        )

        features = self.cnn(x)

        features = features.view(
            batch_size,
            num_frames,
            -1
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

        context = self.dropout(
            context
        )

        output = self.fc(
            context
        )

        return output


device = torch.device("cpu")

model = CNNLSTMAttention(
    num_classes=5,
    hidden_size=256
)

checkpoint_path = (
    "models/best_cnn_lstm_attention.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("=" * 50)
print("ACTIVITY MODEL LOADED SUCCESSFULLY")
print("=" * 50)

print("Device:", device)
print("Classes:", [
    "fall",
    "run",
    "sit",
    "stand",
    "walk"
])

# Test with dummy input
x = torch.randn(
    1,
    16,
    3,
    224,
    224
)

with torch.no_grad():
    output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Model test: SUCCESS")