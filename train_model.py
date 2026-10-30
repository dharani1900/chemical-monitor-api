import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
MODEL_DIR = os.path.join(PROJECT_DIR, "storage", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_data_loaders(batch_size=8):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")
    test_dir = os.path.join(DATASET_DIR, "test")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Dataset directories missing at {DATASET_DIR}")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform) if os.path.exists(test_dir) else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None

    return train_loader, val_loader, test_loader, train_dataset.classes

def build_mobilenet_model(num_classes=2):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Freeze earlier layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze last classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return model

def train_and_evaluate(epochs=5, batch_size=8, lr=0.001, status_callback=None):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_loader, val_loader, test_loader, class_names = get_data_loaders(batch_size)
    model = build_mobilenet_model(num_classes=len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
    info_path = os.path.join(MODEL_DIR, "model_info.json")

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = float(correct_train) / total_train

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = float(correct_val) / total_val

        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        if status_callback:
            status_callback(epoch, epochs, epoch_train_loss, epoch_val_loss, epoch_val_acc)

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc, 4)
        })

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)

    # Test Evaluation
    eval_loader = test_loader if test_loader else val_loader
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    prec = float(precision_score(all_labels, all_preds, average="weighted", zero_division=0))
    rec = float(recall_score(all_labels, all_preds, average="weighted", zero_division=0))
    f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    cm = confusion_matrix(all_labels, all_preds).tolist()

    metrics = {
        "status": "trained",
        "model_version": "MobileNetV2-Custom-v1.0",
        "class_names": class_names,
        "best_val_accuracy": round(best_val_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "history": history,
        "last_trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(info_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training finished successfully!")
    return metrics

if __name__ == "__main__":
    from generate_demo_dataset import generate_demo_dataset
    if not os.path.exists(os.path.join(DATASET_DIR, "train")):
        generate_demo_dataset()
    train_and_evaluate(epochs=5)
