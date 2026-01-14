import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os

# ===============================
# PATH SETUP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ===============================
# DATA TRANSFORMS

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =============================
train_data = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "train"),
    transform=transform
)
val_data = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "val"),
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=16, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=16, shuffle=False
)

num_classes = len(train_data.classes)
print("[INFO] Classes:", train_data.classes)

# ===============================
# MODEL: MobileNetV3-Small

model = models.mobilenet_v3_small(weights="DEFAULT")
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    num_classes
)
model = model.to(device)

# ===============================
# LOSS & OPTIMIZER

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===============================
# TRAINING LOOP

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {avg_train_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

# ===============================
# SAVE MODEL

model_path = os.path.join(MODEL_DIR, "face_classifier.pth")
torch.save(model.state_dict(), model_path)

print(f"[DONE] Model saved at: {model_path}")
