# sffn_qat/test_qat_pytorch.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import pandas as pd
import os
import sys
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torcheval.metrics import PeakSignalNoiseRatio 

# ### ADAPTED ###
# Add the parent directory to the path to import from 'preprocessing'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ### ADAPTED ###
try:
    from preprocessing.dataset import SFFNDataset
except ImportError:
    print("Error: Could not import SFFNDataset. Make sure you are running from sffn_qat/")
    sys.exit(1)

# ### QAT ###
# Import the new QAT model definition
from sffn_qat_model import SFFN_QAT

def test_pytorch_model(checkpoint_path, args):
    """
    Loads and evaluates the trained PyTorch QAT model (.pth).
    This is your "source of truth" for accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing PyTorch QAT model on {device}")

    # --- 1. Load Model Definition and Weights ---
    # ### QAT ###
    model = SFFN_QAT(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # --- CRITICAL: Set model to eval() mode ---
    model.eval()

    # --- 2. Dataloaders (Using your SFFNDataset) ---
    # ### ADAPTED ###
    print("Loading test data...")
    test_df = pd.read_csv(args.test_csv)
    test_dataset = SFFNDataset(
        root_dir=args.dataset,
        df=test_df,
        spatial_transform=None, # Add your transforms here
        freq_transform=None,    # Add your transforms here
        mode='test' 
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )

    # --- 3. Evaluation Loop ---
    all_labels = []
    all_preds_scores = [] # For AUC
    all_preds_classes = [] # For Accuracy/F1
    
    psnr_metric = PeakSignalNoiseRatio(device=device)
    criterion_recon = nn.L1Loss() # For comparison, use same as training

    print("Running evaluation on PyTorch QAT model...")
    with torch.no_grad(): # Disable gradient calculation
        for data, labels in tqdm(test_loader):
            
            # ### ADAPTED ###
            x_spatial = data['spatial'].to(device)
            x_freq = data['freq'].to(device)
            recon_target = data['freq_original'].to(device)
            labels = labels.to(device)
            
            # Forward pass
            class_out, recon_out = model(x_spatial, x_freq)
            
            # Update metrics
            psnr_metric.update(recon_out, recon_target)
            
            scores = torch.softmax(class_out, dim=1)[:, 1] # Probability of class 1
            preds = torch.argmax(class_out, dim=1)
            
            all_labels.append(labels.cpu())
            all_preds_scores.append(scores.cpu())
            all_preds_classes.append(preds.cpu())

    # --- 4. Calculate and Print Final Metrics ---
    # ### ADAPTED ### - Using your project's metrics
    all_labels = torch.cat(all_labels).numpy()
    all_preds_scores = torch.cat(all_preds_scores).numpy()
    all_preds_classes = torch.cat(all_preds_classes).numpy()
    
    final_psnr = psnr_metric.compute().item()
    final_acc = accuracy_score(all_labels, all_preds_classes)
    final_f1 = f1_score(all_labels, all_preds_classes)
    final_auc = roc_auc_score(all_labels, all_preds_scores)

    print("\n--- PyTorch QAT Model Results ---")
    print(f"  Accuracy: {final_acc * 100:.2f}%")
    print(f"  AUC Score: {final_auc:.4f}")
    print(f"  F1 Score: {final_f1:.4f}")
    print(f"  Recon PSNR: {final_psnr:.2f} dB")
    print("-----------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PyTorch QAT Model')
    # ### QAT ###
    parser.add_argument('--qat_checkpoint', type=str, default='sffn_qat_mixed_best.pth',
                        help='Path to the trained QAT model checkpoint.')
    
    # ### ADAPTED ###
    parser.add_argument('--dataset', type=str, default='../dataset/df',
                        help='Path to the dataset directory')
    parser.add.argument('--test_csv', type=str, default='../dataset/test.csv',
                        help='Path to the test CSV file')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Test batch size.')
    args = parser.parse_args()
    
    test_pytorch_model(args.qat_checkpoint, args)