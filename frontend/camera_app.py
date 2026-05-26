import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

st.set_page_config(page_title="Real-time Emotion", layout="centered")

st.title("🎥 实时表情识别系统")

# ===== 类别 =====
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ===== 模型 =====
@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load("../models/emotion.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ===== 预处理 =====
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# ===== 摄像头 =====
run = st.checkbox("开启摄像头")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("无法打开摄像头")
        break

    # BGR → RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    # 预处理
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]

    pred = classes[np.argmax(prob)]

    # 显示结果
    cv2.putText(frame, f"{pred}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(frame)

cap.release()