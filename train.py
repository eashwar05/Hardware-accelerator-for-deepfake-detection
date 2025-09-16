# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from meso_inception4 import MesoInception4
from torchvision import transforms, datasets
from tqdm import tqdm

def train_model():
    """
    Trains the MesoInception-4 model on the CIFAR-10 dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MesoInception4().to(device)
    
    # Define transformations for CIFAR-10
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  # <-- Resize images to fit model input
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Use CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 10 
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), 'mesoinception4_cifar10.pth')
    print("Training finished and model saved.")

if __name__ == '__main__':
    train_model()