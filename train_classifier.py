import os
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.utils.data import Subset
import random

def main():
    DATA_DIR   = Path("dataset/flattened_dataset")
    SAVE_PATH  = Path("models/material_classifier_resnet18.pth")
    BATCH_SIZE = 16
    EPOCHS     = 3
    LR         = 1e-3
    VAL_SPLIT  = 0.2
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 0


    train_tfms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])


    image_folder_ds = datasets.ImageFolder(DATA_DIR, transform=train_tfms)
    class_names = image_folder_ds.classes
    num_classes = len(class_names)


    SUBSET_SIZE = 3000
    subset_indices = random.sample(range(len(image_folder_ds)), SUBSET_SIZE)
    full_ds = Subset(image_folder_ds, subset_indices)


    val_size = int(VAL_SPLIT * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])


    val_ds.dataset.transform = val_tfms


    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Classes: {class_names}")


    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0


    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct = 0.0, 0

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (preds.argmax(1) == labels).sum().item()

        train_acc = 100 * train_correct / train_size


        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs)
                val_loss += criterion(preds, labels).item() * imgs.size(0)
                val_correct += (preds.argmax(1) == labels).sum().item()

        val_acc = 100 * val_correct / val_size

        print(f"[{epoch:02}/{EPOCHS}]  "
              f"train loss {train_loss/train_size:.4f}  acc {train_acc:.2f}% | "
              f"val loss {val_loss/val_size:.4f}  acc {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved new best model @ {best_val_acc:.2f}%")

    print("\nTraining complete. Best val acc:", best_val_acc)

    def predict_image(img_path):
        tfms = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        img = tfms(datasets.folder.default_loader(img_path)).unsqueeze(0).to(DEVICE)

        model.eval()
        with torch.no_grad():
            out = model(img)
            prob = torch.softmax(out, 1)[0]
            idx = prob.argmax().item()
        return class_names[idx], prob[idx].item()

    test_img = "sample_product_images/sampleproduct_plastic.jpg"
    if os.path.isfile(test_img) and os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        label, confidence = predict_image(test_img)
        print(f"{test_img} → {label} ({confidence:.2%})")


if __name__ == "__main__":
    main()
