import io
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

try:
    import onnxruntime as ort
except ImportError:
    ort = None

st.set_page_config(page_title="Emotion AI System", layout="centered")
st.title("😃 AI 表情识别系统（图片 / 摄像头 / 视频）")

classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_PT_MODEL_PATH = PROJECT_ROOT / "models" / "emotion.pt"

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


@st.cache_resource
def load_default_pt_model():
    if not DEFAULT_PT_MODEL_PATH.exists():
        raise FileNotFoundError(f"默认模型不存在：{DEFAULT_PT_MODEL_PATH}")

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(str(DEFAULT_PT_MODEL_PATH), map_location="cpu"))
    model.eval()
    return model.to(device)


@st.cache_resource
def load_pt_model_from_bytes(model_bytes: bytes):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    state_dict = torch.load(io.BytesIO(model_bytes), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


@st.cache_resource
def load_onnx_session_from_bytes(model_bytes: bytes):
    if ort is None:
        raise RuntimeError("当前环境未安装 onnxruntime，无法加载 ONNX 模型。")

    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / "uploaded_emotion_model.onnx"
    temp_path.write_bytes(model_bytes)

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    return ort.InferenceSession(str(temp_path), providers=providers)


def get_device_label(model_type: str, onnx_session=None):
    if model_type == "onnx":
        if onnx_session is None:
            return "onnx (未加载)"
        return f"onnx ({', '.join(onnx_session.get_providers())})"

    if device.type == "cuda":
        return f"pytorch cuda ({torch.cuda.get_device_name(0)})"
    return "pytorch cpu"


def resize_frame_for_display(frame, max_width):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    ratio = max_width / width
    new_size = (int(width * ratio), int(height * ratio))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def preprocess_image(img):
    tensor = transform(img).unsqueeze(0)
    return tensor


def predict_with_pt(img, model):
    input_tensor = preprocess_image(img).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
    return prob


def predict_with_onnx(img, session):
    input_tensor = preprocess_image(img).numpy().astype(np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0][0]
    exp_output = np.exp(output - np.max(output))
    prob = exp_output / np.sum(exp_output)
    return prob


def predict(img, model_type, pt_model=None, onnx_session=None):
    if model_type == "onnx":
        return predict_with_onnx(img, onnx_session)
    return predict_with_pt(img, pt_model)


def draw_prediction(frame, pred, confidence, fps=None):
    lines = [f"Emotion: {pred}", f"Conf: {confidence:.2f}"]
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")

    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (20, 40 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    return frame


def run_video_stream(
    cap,
    frame_placeholder,
    model_type,
    pt_model=None,
    onnx_session=None,
    target_fps=12,
    infer_every_n_frames=3,
    display_width=640,
):
    frame_interval = 1.0 / target_fps
    frame_count = 0
    last_pred = "detecting..."
    last_conf = 0.0
    fps_window_start = time.time()
    fps_frame_counter = 0
    current_fps = 0.0

    while cap.isOpened():
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        fps_frame_counter += 1

        now = time.time()
        elapsed = now - fps_window_start
        if elapsed >= 1.0:
            current_fps = fps_frame_counter / elapsed
            fps_window_start = now
            fps_frame_counter = 0

        if frame_count == 1 or frame_count % infer_every_n_frames == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            prob = predict(pil, model_type, pt_model, onnx_session)
            pred_index = int(np.argmax(prob))
            last_pred = classes[pred_index]
            last_conf = float(prob[pred_index])

        annotated_frame = draw_prediction(frame.copy(), last_pred, last_conf, current_fps)
        annotated_frame = resize_frame_for_display(annotated_frame, display_width)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_frame, channels="RGB")

        processing_time = time.time() - loop_start
        if processing_time < frame_interval:
            time.sleep(frame_interval - processing_time)

    cap.release()


st.sidebar.header("模型设置")
model_source = st.sidebar.radio("模型来源", ["默认 PT 模型", "上传模型文件"])

selected_model_type = "pt"
pt_model = None
onnx_session = None

if model_source == "默认 PT 模型":
    try:
        pt_model = load_default_pt_model()
    except FileNotFoundError as exc:
        st.sidebar.error(str(exc))
else:
    uploaded_model = st.sidebar.file_uploader("上传模型", type=["pt", "onnx"])
    if uploaded_model is not None:
        model_bytes = uploaded_model.read()
        suffix = Path(uploaded_model.name).suffix.lower()

        if suffix == ".pt":
            pt_model = load_pt_model_from_bytes(model_bytes)
            selected_model_type = "pt"
            st.sidebar.success("已加载 PT 模型")
        elif suffix == ".onnx":
            selected_model_type = "onnx"
            try:
                onnx_session = load_onnx_session_from_bytes(model_bytes)
                st.sidebar.success("已加载 ONNX 模型")
            except RuntimeError as exc:
                st.sidebar.error(str(exc))
        else:
            st.sidebar.error("仅支持 .pt 或 .onnx 模型文件")

if model_source == "上传模型文件" and pt_model is None and onnx_session is None:
    st.warning("请先上传一个 .pt 或 .onnx 模型文件。")

st.caption(f"当前推理设备：{get_device_label(selected_model_type, onnx_session)}")
mode = st.radio("选择模式", ["图片识别", "摄像头识别", "视频识别"])

can_infer = pt_model is not None or onnx_session is not None

if mode == "图片识别":
    file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

    if file and can_infer:
        img = Image.open(file)
        st.image(img, caption="输入图片")

        prob = predict(img, selected_model_type, pt_model, onnx_session)
        pred = classes[int(np.argmax(prob))]

        st.success(f"预测结果：{pred}")

        for i, c in enumerate(classes):
            st.write(f"{c}: {prob[i]:.2f}")
            st.progress(float(prob[i]))

elif mode == "摄像头识别":
    target_fps = st.slider("显示帧率", min_value=5, max_value=20, value=15, step=1)
    infer_every_n_frames = st.slider("每隔几帧识别一次", min_value=1, max_value=5, value=2, step=1)
    display_width = st.select_slider("显示分辨率宽度", options=[480, 640, 800, 960], value=640)
    run = st.checkbox("开启摄像头")

    frame_placeholder = st.image([])

    if run and can_infer:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("无法打开摄像头")
        else:
            run_video_stream(
                cap,
                frame_placeholder,
                selected_model_type,
                pt_model,
                onnx_session,
                target_fps,
                infer_every_n_frames,
                display_width,
            )

elif mode == "视频识别":
    target_fps = st.slider("播放帧率", min_value=5, max_value=24, value=15, step=1)
    infer_every_n_frames = st.slider("每隔几帧识别一次", min_value=1, max_value=5, value=2, step=1)
    display_width = st.select_slider("显示分辨率宽度", options=[480, 640, 800, 960], value=640)
    file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])

    if file and can_infer:
        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / file.name
        temp_path.write_bytes(file.read())

        cap = cv2.VideoCapture(str(temp_path))
        frame_placeholder = st.image([])

        if not cap.isOpened():
            st.error("无法打开视频文件")
        else:
            run_video_stream(
                cap,
                frame_placeholder,
                selected_model_type,
                pt_model,
                onnx_session,
                target_fps,
                infer_every_n_frames,
                display_width,
            )
