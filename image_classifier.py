import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import time

class UniversalImageDataset(Dataset):
    def __init__(self, image_size=224, is_color=True):
        current_dir = os.getcwd()
        self.img_dir = os.path.join(current_dir, 'data', 'images')
        labels_file = os.path.join(current_dir, 'data', 'labels.csv')
        
        print("\n1. Loading Dataset")
        print("-" * 50)
        print(f"Loading images from: {self.img_dir}")
        
        self.labels = pd.read_csv(labels_file)
        print("\nDataset Contents:")
        print(self.labels)
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = self.transform(image)
        label = self.labels.iloc[idx, 1]
        return image, label

class ModernNet(nn.Module):
    def __init__(self, num_classes):
        super(ModernNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 56 * 56, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_model(model, trainloader, testloader, num_epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n3. Training Progress")
    print("-" * 50)
    print(f"Using device: {device}")
    print(f"Number of epochs: {num_epochs}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 2 == 1:
                print(f"Batch {i+1:3d} | Loss: {running_loss/2:.3f} | "
                      f"Training Accuracy: {100.*correct/total:.1f}%")
                running_loss = 0.0
        
        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f"Epoch {epoch + 1} Test Accuracy: {test_acc:.1f}%")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model saved as 'trained_model.pth'")

def main():
    print("\nImage Classification Training")
    print("=" * 50)
    
    # Create dataset
    dataset = UniversalImageDataset()
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    print("\n2. Dataset Split")
    print("-" * 50)
    print(f"Training samples: {train_size}")
    print(f"Testing samples: {test_size}")
    print(f"Number of classes: {len(dataset.labels['class_label'].unique())}")
    
    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=3, shuffle=True)
    testloader = DataLoader(testset, batch_size=3, shuffle=False)
    
    # Create and train model
    model = ModernNet(num_classes=len(dataset.labels['class_label'].unique()))
    train_model(model, trainloader, testloader)

if __name__ == '__main__':
    main()