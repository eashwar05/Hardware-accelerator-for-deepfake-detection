# evaluate.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from meso_inception4 import MesoInception4
from torchvision import transforms, datasets
import time

def evaluate_model(model_path, data_dir, device):
    """
    Evaluates the trained model on the CIFAR-10 test set.
    """
    model = MesoInception4()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    transforms_ = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the CIFAR-10 test set
    dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms_)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    all_labels = []
    all_preds = []
    
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get the predicted class (the one with the highest score)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    end_time = time.time()

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    conf_mat = confusion_matrix(all_labels, all_preds)

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total inference time: {end_time - start_time:.2f} seconds")
    print(f"Average FPS: {len(dataset) / (end_time - start_time):.2f}")
    
    print("\nConfusion Matrix:")
    print(conf_mat)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # You need a trained model file from the training script
    evaluate_model('mesoinception4_cifar10.pth', 'data', device)