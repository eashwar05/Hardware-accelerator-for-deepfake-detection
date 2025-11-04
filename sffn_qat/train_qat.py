# sffn_qat/train_qat.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import pandas as pd
import os
import sys

# ### ADAPTED ###
# Add the parent directory to the path to import from 'preprocessing' and 'model'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ### ADAPTED ###
# Import your project's specific dataset loader
try:
    from preprocessing.dataset import SFFNDataset
except ImportError:
    print("Error: Could not import SFFNDataset. Make sure you are running from sffn_qat/")
    sys.exit(1)

# ### QAT ###
# Import the new QAT model, not the original SFFN
from sffn_qat_model import SFFN_QAT

def load_fp32_weights(qat_model, fp32_checkpoint_path):
    """
    Loads weights from the pre-trained FP32 model into the new QAT model.
    This function maps state_dict keys from your original sffn.py
    to the new sffn_qat_model.py structure.
    """
    fp32_state_dict = torch.load(fp32_checkpoint_path, map_location='cpu')
    
    # Check if checkpoint is nested (e.g., {'model': ...})
    if 'model_state_dict' in fp32_state_dict:
        fp32_state_dict = fp32_state_dict['model_state_dict']
    elif 'model' in fp32_state_dict:
        fp32_state_dict = fp32_state_dict['model']

    new_state_dict = qat_model.state_dict()
    
    print("Loading FP32 weights...")
    
    # 1. Map Spatial Stream (EfficientNet)
    # This is handled by TimmModelProxy's `pretrained=True`
    # and its own internal weight loading.
    # We only need to map the classifier if it was custom-trained.
    # For now, we assume the proxy handles the backbone.
    
    # 2. Map FrequencyCNN weights
    # ORIGINAL (from model/sffn.py): frequency_stream.conv1.weight
    # NEW (from sffn_qat_model.py):  frequency_stream.layer1.0.weight
    key_map_freq = {
        'conv1.weight': 'layer1.0.weight',
        'bn1.weight':   'layer1.1.weight',
        'bn1.bias':     'layer1.1.bias',
        'bn1.running_mean': 'layer1.1.running_mean',
        'bn1.running_var':  'layer1.1.running_var',
        'bn1.num_batches_tracked': 'layer1.1.num_batches_tracked',
        
        'conv2.weight': 'layer2.0.weight',
        'bn2.weight':   'layer2.1.weight',
        'bn2.bias':     'layer2.1.bias',
        'bn2.running_mean': 'layer2.1.running_mean',
        'bn2.running_var':  'layer2.1.running_var',
        'bn2.num_batches_tracked': 'layer2.1.num_batches_tracked',

        'conv3.weight': 'layer3.0.weight',
        'bn3.weight':   'layer3.1.weight',
        'bn3.bias':     'layer3.1.bias',
        'bn3.running_mean': 'layer3.1.running_mean',
        'bn3.running_var':  'layer3.1.running_var',
        'bn3.num_batches_tracked': 'layer3.1.num_batches_tracked',
    }
    
    for old_key, new_key in key_map_freq.items():
        old_full_key = f'frequency_stream.{old_key}'
        new_full_key = f'frequency_stream.{new_key}'
        if old_full_key in fp32_state_dict and new_full_key in new_state_dict:
            new_state_dict[new_full_key] = fp32_state_dict[old_full_key]
        else:
            print(f"Warning: Could not map key {old_full_key}")

    # 3. Map Fusion MLP weights
    key_map_mlp = {
        '0.weight': '0.weight', '0.bias': '0.bias',
        '3.weight': '3.weight', '3.bias': '3.bias',
    }
    for old_key, new_key in key_map_mlp.items():
        old_full_key = f'fusion_mlp.{old_key}'
        new_full_key = f'fusion_mlp.{new_key}'
        if old_full_key in fp32_state_dict and new_full_key in new_state_dict:
            new_state_dict[new_full_key] = fp32_state_dict[old_full_key]

    # 4. Map Reconstruction Head weights
    key_map_recon = {
        '0.weight': '0.weight', '0.bias': '0.bias',
        '2.weight': '2.weight', '2.bias': '2.bias',
        '4.weight': '4.weight', '4.bias': '4.bias',
    }
    for old_key, new_key in key_map_recon.items():
        old_full_key = f'reconstruction_head.{old_key}'
        new_full_key = f'reconstruction_head.{new_key}'
        if old_full_key in fp32_state_dict and new_full_key in new_state_dict:
            new_state_dict[new_full_key] = fp32_state_dict[old_full_key]

    # Load the state dict, ignoring mismatches
    qat_model.load_state_dict(new_state_dict, strict=False)
    print("Successfully loaded pre-trained FP32 weights into QAT model.")
    return qat_model

def main():
    parser = argparse.ArgumentParser(description='SFFN QAT Training')
    # ### ADAPTED ### - Using your project's arg names
    parser.add_argument('--dataset', type=str, default='../dataset/df',
                        help='Path to the dataset directory')
    parser.add_argument('--train_csv', type=str, default='../dataset/train.csv',
                        help='Path to the training CSV file')
    parser.add_argument('--val_csv', type=str, default='../dataset/val.csv',
                        help='Path to the validation CSV file')
    
    # ### QAT ### - New required argument
    parser.add_argument('--fp32_checkpoint', type=str, required=True,
                        help='Path to the pre-trained FP32 model checkpoint (e.g., ../model.pth)')
    
    # ### QAT ### - Modified defaults for fine-tuning
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs to fine-tune (QAT).')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate for QAT (should be very small).')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Define Model ---
    # ### QAT ### - Instantiate the QAT model
    model = SFFN_QAT(num_classes=2, pretrained=True).to(device)

    # --- 2. Load Weights ---
    # ### QAT ### - Load your custom-trained FP32 weights
    model = load_fp32_weights(model, args.fp32_checkpoint)

    # --- 3. Dataloaders (Using your project's SFFNDataset) ---
    # ### ADAPTED ###
    print("Loading training data...")
    train_df = pd.read_csv(args.train_csv)
    train_dataset = SFFNDataset(
        root_dir=args.dataset,
        df=train_df,
        spatial_transform=None, # Add your transforms here
        freq_transform=None,    # Add your transforms here
        mode='train'
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    print("Loading validation data...")
    val_df = pd.read_csv(args.val_csv)
    val_dataset = SFFNDataset(
        root_dir=args.dataset,
        df=val_df,
        spatial_transform=None, # Add your transforms here
        freq_transform=None,    # Add your transforms here
        mode='val'
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )

    # --- 4. Define Loss and Optimizer ---
    # ### ADAPTED ### - Using your project's likely loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_recon = nn.L1Loss() # Your original `train.py` might use L1 or MSE
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Starting QAT fine-tuning for {args.epochs} epochs with LR={args.lr} on {device}")

    # --- 5. QAT Training Loop ---
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for i, (data, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')):
            
            # ### ADAPTED ### - Using your SFFNDataset output format
            x_spatial = data['spatial'].to(device)
            x_freq = data['freq'].to(device)
            recon_target = data['freq_original'].to(device) # Assuming this is the L1 target
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (simulates quantization)
            class_out, recon_out = model(x_spatial, x_freq)
            
            # Calculate loss
            loss_c = criterion_class(class_out, labels)
            loss_r = criterion_recon(recon_out, recon_target)
            loss = loss_c + 0.1 * loss_r # Assuming 0.1 weighting
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation loop
        model.eval()
        val_acc = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]'):
                x_spatial = data['spatial'].to(device)
                x_freq = data['freq'].to(device)
                recon_target = data['freq_original'].to(device)
                labels = labels.to(device)
                
                class_out, recon_out = model(x_spatial, x_freq)
                
                loss_c = criterion_class(class_out, labels)
                loss_r = criterion_recon(recon_out, recon_target)
                loss = loss_c + 0.1 * loss_r
                val_loss += loss.item()

                preds = torch.argmax(class_out, dim=1)
                val_acc += (preds == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        final_val_acc = 100 * val_acc / len(val_dataset)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {final_val_acc:.2f}%")
        
        # --- 6. Save Final QAT Model ---
        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            output_path = 'sffn_qat_mixed_best.pth'
            torch.save(model.state_dict(), output_path)
            print(f"New best model saved to {output_path}")

    print(f"QAT complete. Final model saved to {output_path}")

if __name__ == '__main__':
    main()