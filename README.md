# Emotion Recognition System

基于 PyTorch 的表情识别系统课程作业项目，完整覆盖了深度学习任务中的数据处理、模型训练、模型导出、前端推理与可视化展示流程。

本项目的任务方向属于**图像分类**，场景为**人脸表情识别**，支持：
- 图片识别
- 摄像头实时识别
- 视频文件识别
- 前端导入 `.pt` / `.onnx` 模型进行推理

---

## 1. 项目功能

本系统主要实现以下功能：

1. 基于 PyTorch 训练表情识别模型
2. 使用 `ImageFolder` 方式组织和读取数据集
3. 将训练好的模型导出为：
   - PyTorch 模型：`.pt`
   - ONNX 模型：`.onnx`
4. 使用 Streamlit 构建可交互前端页面
5. 前端支持三种输入模式：
   - 图片识别
   - 摄像头识别
   - 视频识别
6. 前端支持两种模型导入方式：
   - 默认加载本地 `.pt` 模型
   - 手动上传 `.pt` 或 `.onnx` 模型文件
7. 支持推理结果可视化：
   - 分类结果显示
   - 概率显示
   - 视频帧率显示
   - 当前推理设备显示

---

## 2. 项目结构

```text
emotion-recognition/
├─ backend/
│  ├─ inference_onnx.py
│  └─ inference_pt.py
├─ data/
│  ├─ train/
│  │  ├─ angry/
│  │  ├─ disgust/
│  │  ├─ fear/
│  │  ├─ happy/
│  │  ├─ neutral/
│  │  ├─ sad/
│  │  └─ surprise/
│  └─ test/
│     ├─ angry/
│     ├─ disgust/
│     ├─ fear/
│     ├─ happy/
│     ├─ neutral/
│     ├─ sad/
│     └─ surprise/
├─ frontend/
│  └─ app.py
├─ models/
│  ├─ emotion.pt
│  └─ emotion.onnx
├─ train/
│  └─ train.py
└─ README.md
```

---

## 3. 数据集组织方式

本项目使用 `torchvision.datasets.ImageFolder` 读取数据，因此数据集目录需要按类别分文件夹存放。

目录结构如下：

```text
data/
  train/
    angry/
    happy/
    sad/
    surprise/
    neutral/
    fear/
    disgust/

  test/
    angry/
    happy/
    sad/
    surprise/
    neutral/
    fear/
    disgust/
```

每个类别文件夹中存放对应表情类别的图片。

---

## 4. 使用技术栈

- Python
- PyTorch
- Torchvision
- OpenCV
- Streamlit
- NumPy
- Pillow
- Matplotlib
- ONNX
- ONNX Runtime（可选，用于 ONNX 推理）

---

## 5. 环境安装

建议使用 Python 3.9 及以上版本。

### 安装依赖

```bash
pip install torch torchvision streamlit opencv-python pillow numpy matplotlib onnx onnxruntime
```

如果需要使用 GPU 版 ONNX Runtime，可安装：

```bash
pip install onnxruntime-gpu
```

---

## 6. 模型训练

训练脚本位置：

```text
train/train.py
```

训练流程包括：
- 图像预处理
- 构建 `ImageFolder` 数据集
- 构建 `DataLoader`
- 加载 ResNet18 模型
- 修改最后全连接层输出类别数
- 模型训练
- 保存 `.pt` 模型
- 导出 `.onnx` 模型
- 绘制训练损失与准确率曲线

### 运行训练

```bash
cd train
python train.py
```

训练完成后会在 `models/` 目录下生成：

- `emotion.pt`
- `emotion.onnx`

如果已创建结果目录，也会在 `results/` 下生成：
- `loss.png`
- `acc.png`

---

## 7. 前端推理与系统运行

前端文件位置：

```text
frontend/app.py
```

### 启动前端

```bash
cd frontend
streamlit run app.py
```

启动后可在浏览器中使用以下功能：

### 1）图片识别
- 上传图片文件
- 输出情感分类结果
- 显示各类别概率

### 2）摄像头识别
- 实时调用本机摄像头
- 支持帧率控制
- 支持跳帧推理优化
- 支持显示分辨率缩放
- 显示当前 FPS

### 3）视频识别
- 上传本地视频文件
- 逐帧识别视频中的表情类别
- 支持帧率控制和流畅度优化

---

## 8. 前端导入模型说明

为满足课程作业“前端可导入 `.pt` 或 `.onnx` 模型”的要求，本项目前端支持以下两种方式：

### 方式一：默认加载本地 PT 模型
前端自动读取：

```text
models/emotion.pt
```

### 方式二：手动上传模型文件
在左侧边栏中可选择：
- 上传 `.pt` 模型
- 上传 `.onnx` 模型

系统会根据模型后缀自动选择推理方式：
- `.pt` → 使用 PyTorch 推理
- `.onnx` → 使用 ONNX Runtime 推理

说明：
- 当前 `.pt` 模型默认适配 ResNet18 + 7 分类结构
- 当前 `.onnx` 模型默认输入尺寸为 `1 x 3 x 48 x 48`

---

## 9. 可视化与优化说明

本项目包含以下可视化与性能优化内容：

### 可视化内容
- 图片识别结果展示
- 分类概率展示
- 视频帧推理结果叠加显示
- FPS 显示
- 当前推理设备显示

### 性能优化内容
- 跳帧推理
- 帧率控制
- 视频显示分辨率缩放
- 摄像头/视频流畅度优化

---

## 10. 预训练模型与参考出处

本项目训练时使用了 `torchvision` 提供的 **ResNet18 预训练模型** 作为基础网络。

### 预训练模型来源
- Torchvision 官方模型库：
  [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)

### ResNet 论文出处
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
  **Deep Residual Learning for Image Recognition**.
  CVPR 2016.
  [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

如果使用的数据集、公开视频、展示图片来源于第三方，请在提交作业时补充其具体来源。

---

## 11. 作业提交材料建议

为满足课程要求，建议最终提交以下材料：

1. 项目源码
2. `README.md`
3. 训练好的模型文件：
   - `emotion.pt`
   - `emotion.onnx`
4. GitHub 或 Gitee 仓库链接 / 截图
5. 项目演示 PPT
6. 项目运行演示视频
7. 系统运行截图

---

## 12. 合规声明

本项目用于深度学习课程作业展示与教学实践，内容主要围绕人脸表情识别，不包含违法、违规、暴力、色情、政治敏感或其他不当信息。

使用者应保证：
- 数据来源合法合规
- 不将本项目用于非法用途
- 不侵犯他人隐私、肖像权或其他合法权益

---

## 13. 后续可优化方向

后续还可继续补充以下功能：

- 人脸检测后再进行表情识别
- 使用 ONNX Runtime / TensorRT 进一步提速
- 增加更多类别或更高精度模型
- 增加后端接口封装，形成前后端分离架构
- 完善训练评估指标，如混淆矩阵、Precision、Recall、F1-score

---

## 14. 作者说明

项目仓库链接：https://github.com/cloriseeun-hyt/emotion-recognition

本项目为课程作业项目，基于 PyTorch 完成图像分类方向的人脸表情识别系统实现。

如用于课程答辩，建议配合：
- 运行截图
- 模型效果展示
- 训练曲线图
- 仓库链接
- 演示视频
- PPT 讲解材料
