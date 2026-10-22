# Goal: road scene segmentation
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

NUM_CLASSES = 5
CLASSES = ["road", "car", "person", "building", "sky", "bicycle", "sign"]

image_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(32, 7, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.head(x)


train_set = RoadScenes("data/train", transform=image_tf)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

model = SegNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(50):
    model.train()
    for images, masks in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
