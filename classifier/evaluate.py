import torch
from torchvision import datasets, transforms, models
from torch import nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# ===============================
# PATH SETUP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_classifier.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===============================
# LOAD TEST DATA

test_data = datasets.ImageFolder(DATASET_DIR, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=16, shuffle=False
)

class_names = test_data.classes
num_classes = len(class_names)

# ===============================

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    num_classes
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ===============================
# EVALUATION

y_true = []
y_pred = []

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"[RESULT] Test Accuracy: {test_accuracy:.2f}%")

# ===============================
# CONFUSION MATRIX

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Face Classification Confusion Matrix")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

print("[DONE] Confusion matrix saved to results/confusion_matrix.png")
