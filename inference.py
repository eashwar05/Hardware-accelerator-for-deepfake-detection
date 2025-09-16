# inference.py
import torch
import time
from PIL import Image
from torchvision import transforms
from meso_inception4 import MesoInception4

def predict(model_path, image_path, device):
    """
    Loads a model and performs inference on a single image.
    """
    model = MesoInception4()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    inference_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = inference_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)
        end_time = time.time()
        
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob > 0.5 else 0
        
        print(f"Image: {image_path}")
        print(f"Prediction: {'FAKE' if prediction == 1 else 'REAL'} (Confidence: {prob:.4f})")
        print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
        
    return prediction, prob

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # You'll need to provide a trained model and a test image.
    # e.g., predict('mesoinception4_baseline.pth', 'data/test_image.png', device)
    print("Please provide a trained model and a test image to run this script.")