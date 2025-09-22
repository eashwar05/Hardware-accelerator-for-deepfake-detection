import torch, argparse, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from model.sffn import SFFN
from utils.dataset import FaceDataset
from utils.transforms import get_transforms

def evaluate_model(data_dir, model_path, batch_size=32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    spatial_transform, freq_transform = get_transforms()
    test_dataset = FaceDataset(data_dir, spatial_transform, freq_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SFFN(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for s_tensor, f_tensor, labels in test_loader:
            s_tensor, f_tensor = s_tensor.to(device), f_tensor.to(device)
            cls_output, _ = model(s_tensor, f_tensor)
            probs = torch.nn.functional.softmax(cls_output, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy()); all_preds.extend(torch.max(cls_output, 1)[1].cpu().numpy()); all_probs.extend(probs.cpu().numpy())
    
    print("--- Evaluation Results ---")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(all_labels, all_probs):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the SFFN model."); parser.add_argument('--data_dir', required=True); parser.add_argument('--model_path', required=True)
    args = parser.parse_args()
    evaluate_model(args.data_dir, args.model_path)