import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ===== 数据处理 =====
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

train_dir = "../data/train"
test_dir = "../data/test"

train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

classes = train_data.classes
num_classes = len(classes)
print("Classes:", classes)

# ===== 模型 =====
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===== 记录 =====
train_losses = []
train_accs = []

# ===== 训练 =====
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    train_losses.append(total_loss)
    train_accs.append(acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f} Acc: {acc:.2f}%")

# ===== 保存模型 =====
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/emotion.pt")

# ===== 画图 =====
plt.figure()
plt.plot(train_losses)
plt.title("Loss Curve")
plt.savefig("../results/loss.png")

plt.figure()
plt.plot(train_accs)
plt.title("Accuracy Curve")
plt.savefig("../results/acc.png")

print("Training finished + model saved")

# ===== 保存 PyTorch 模型 =====
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/emotion.pt")

print("PyTorch model saved!")

# ===== ONNX 导出（关键）=====
model.eval()  # ⚠️必须加（很重要）

dummy = torch.randn(1, 3, 48, 48).to(DEVICE)

torch.onnx.export(
    model,
    dummy,
    "../models/emotion.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("ONNX exported!")