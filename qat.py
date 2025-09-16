# qat.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from meso_inception4 import MesoInception4
from train import FaceForensicsDataset
from torchvision import transforms
import os
from tqdm import tqdm

def train_qat_model():
    """
    Performs Quantization-Aware Training (QAT) to fine-tune the model for int8.
    """
    device = torch.device("cpu")
    
    # 1. Load the pre-trained full-precision model
    model = MesoInception4()
    model.load_state_dict(torch.load('mesoinception4_baseline.pth', map_location='cpu'))
    model.eval()
    
    # 2. Prepare the model for QAT by inserting fake quantization ops
    torch.backends.quantized.engine = 'qnnpack'
    model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    
    # Fusing layers is a best practice. PyTorch's `prepare_qat` handles this
    # but for some models, you may need to manually fuse `nn.Conv2d` with `nn.BatchNorm2d`
    # and `nn.ReLU`. For this model, we will rely on Vitis AI to handle fusion.
    model_qat = torch.quantization.prepare_qat(model.train(), inplace=False)
    model_qat.to(device)
    
    # 3. Calibration
    print("Running initial calibration for QAT...")
    dummy_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    calibration_dataset = FaceForensicsDataset(root_dir='data/ffhq_cropped_train', transform=dummy_transforms)
    calibration_loader = DataLoader(calibration_dataset, batch_size=16, num_workers=2)
    
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= 10: break # Calibrate on 10 batches
            model_qat(images.to(device))
            
    # 4. QAT Fine-Tuning Loop
    print("Starting QAT fine-tuning...")
    optimizer = optim.Adam(model_qat.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 3
    for epoch in range(epochs):
        model_qat.train()
        running_loss = 0.0
        for inputs, labels in tqdm(calibration_loader, desc=f"QAT Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_qat(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"QAT Epoch {epoch+1}, Loss: {running_loss / len(calibration_loader):.4f}")
        
    # 5. Convert to the final quantized model
    model_qat.eval()
    quantized_model = torch.quantization.convert(model_qat, inplace=True)
    
    # 6. Save the final quantized model state_dict
    torch.save(quantized_model.state_dict(), 'mesoinception4_qat.pth')
    print("QAT fine-tuning finished and quantized model saved.")

if __name__ == '__main__':
    train_qat_model()