import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
import os
import cv2
import numpy as np
from tqdm import tqdm
from src.decision import DrivingCNN, ACTIONS


class DrivingDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

        for label_idx, action in enumerate(ACTIONS):
            action_dir = os.path.join(root_dir, action)
            if os.path.exists(action_dir):
                for img_file in os.listdir(action_dir):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((
                            os.path.join(action_dir, img_file),
                            label_idx
                        ))

        if len(self.samples) == 0:
            raise ValueError(
                "No images found in data/train/\n"
                "Run: python dataset.py first to extract nuScenes data."
            )

        print(f"Loaded {len(self.samples)} images across {len(ACTIONS)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), label


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load full dataset
    full_dataset = DrivingDataset("data/train")

    # ── Weighted sampler to fix class imbalance ──────────────────
    labels_list         = [full_dataset.samples[i][1] for i in range(len(full_dataset))]
    class_counts_actual = [labels_list.count(i) for i in range(len(ACTIONS))]
    print("Class distribution:")
    for i, (action, count) in enumerate(zip(ACTIONS, class_counts_actual)):
        print(f"  {action:15s}: {count} images")

  
    # 80/20 train/val split
    val_size   = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Sampler only on train_set
    train_labels  = [full_dataset.samples[i][1] for i in train_set.indices]
    train_counts  = [train_labels.count(i) for i in range(len(ACTIONS))]
    train_weights = [1.0 / train_counts[l] if train_counts[l] > 0 else 0.0
                     for l in train_labels]
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    # Use sampler instead of shuffle=True
    train_loader = DataLoader(train_set, batch_size=32,
                              sampler=train_sampler, num_workers=2)
    val_loader   = DataLoader(val_set, batch_size=32,
                              shuffle=False, num_workers=2)

    print(f"Train: {train_size} | Val: {val_size}")

    model     = DrivingCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()  # No weight needed — sampler handles balance
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(20):
        # ── Train ──
        model.train()
        total_loss = 0
        correct    = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/20 [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            action_probs, _ = model(images)
            loss = criterion(action_probs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (action_probs.argmax(1) == labels).sum().item()

        train_acc = correct / train_size * 100

        # ── Validate ──
        model.eval()
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                action_probs, _ = model(images)
                val_correct += (action_probs.argmax(1) == labels).sum().item()

        val_acc = val_correct / val_size * 100
        scheduler.step()

        print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/driving_cnn.pth")
            print(f"  💾 Saved best model (val acc: {val_acc:.2f}%)")

    print(f"\n✅ Training complete. Best Val Acc: {best_val_acc:.2f}%")
    print("Model saved → models/driving_cnn.pth")


if __name__ == "__main__":
    train()
