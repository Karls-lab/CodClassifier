import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms.functional import to_tensor
from PIL import Image
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import sys
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms import functional as F
from torchvision.datasets import ImageFolder
import os

# Define the number of classes (including background)
num_classes = 2
backbone = torchvision.models.resnet50(pretrained=True)
num_out_channels = 256
backbone.fc = nn.Linear(backbone.fc.in_features, num_out_channels)
backbone.out_channels = num_out_channels
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
model.train()



class CustomYOLODetection(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [f for f in os.listdir(root) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        ann_path = os.path.join(self.root, self.image_files[idx].replace(".jpg", ".txt"))
        image = Image.open(img_path).convert("RGB")
        with open(ann_path, "r") as file:
            lines = file.read().splitlines()
        # Parse YOLO annotation
        boxes = []
        labels = []
        for line in lines:
            parts = line.split()
            labels.append(float(parts[0]))  # Assuming class is the first element
            x_center, y_center, width, height = map(float, parts[1:])
            xmin, ymin, xmax, ymax = (
                x_center - width / 2,
                y_center - height / 2,
                x_center + width / 2,
                y_center + height / 2,
            )
            boxes.append([xmin, ymin, xmax, ymax])
            target = {"boxes": torch.tensor(boxes, dtype=torch.float32), "category_id": torch.tensor(labels)}
            if self.transform:
                image = self.transform(image)
            return image, target





yolo_root = "../YOLO-Boxes/NormalizedImages/"
transform = Compose([ToTensor()])
yolo_dataset = CustomYOLODetection(root=yolo_root, transform=transform)
data_loader = DataLoader(yolo_dataset, batch_size=32, shuffle=True)



"""
Train the Model
"""
device = torch.device('cuda') 
model.to(device)


# Define Optimizer and Loss Function
criterion = torch.nn.CrossEntropyLoss()
# params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10  # Adjust the number of epochs as needed

for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        print(f'targets: {targets}')
        device = torch.device('cuda')
        images = [image.to(device) for image in images]
        targets = {k: v.to(device) for k, v in targets.items()}


        # sys.exit()

        # Clear previous gradients
        optimizer.zero_grad()

        print(f"SLKDFJSDLKFJSDLKFJ {images[0].shape}")

        # sys.exit()

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        losses.backward()

        # Update weights
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {losses.item()}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')

