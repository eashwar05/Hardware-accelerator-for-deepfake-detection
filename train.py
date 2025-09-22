import torch, torch.optim as optim, argparse, os
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model.sffn import SFFN
from model.loss import SFFNLoss
from utils.dataset import FaceDataset
from utils.transforms import get_transforms

def train_model(data_dir, epochs, batch_size, lr, warmup_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    spatial_transform, freq_transform = get_transforms()
    full_dataset = FaceDataset(data_dir, spatial_transform, freq_transform)
    train_size = len(full_dataset) - int(0.1 * len(full_dataset))
    train_dataset, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SFFN(num_classes=2).to(device)
    criterion = SFFNLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

    best_val_acc = 0.0; os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups: param_group['lr'] = lr * (epoch + 1) / warmup_epochs
        
        if epoch < 15: criterion.mu_recon, criterion.lambda_cls = 1.0, 0.5
        else:
            progress = (epoch - 15) / (epochs - 15); criterion.mu_recon = max(0.1, 1.0 - progress); criterion.lambda_cls = min(1.0, 0.5 + progress)
        
        print(f"Epoch {epoch+1}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.6f} | Lambda_cls: {criterion.lambda_cls:.2f} | Mu_recon: {criterion.mu_recon:.2f}")

        model.train(); train_loss, train_corrects = 0.0, 0
        for s_tensor, f_tensor, labels in tqdm(train_loader, desc="[Train]"):
            s_tensor, f_tensor, labels = s_tensor.to(device), f_tensor.to(device), labels.to(device)
            optimizer.zero_grad()
            cls_out, recon_out = model(s_tensor, f_tensor)
            loss, _, _ = criterion((cls_out, recon_out), (labels, f_tensor))
            loss.backward(); optimizer.step()
            train_loss += loss.item() * s_tensor.size(0)
            train_corrects += torch.sum(torch.max(cls_out, 1)[1] == labels.data)
        print(f'Train Loss: {train_loss / len(train_dataset):.4f} Acc: {train_corrects.double() / len(train_dataset):.4f}')

        model.eval(); val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for s_tensor, f_tensor, labels in tqdm(val_loader, desc="[Val]"):
                s_tensor, f_tensor, labels = s_tensor.to(device), f_tensor.to(device), labels.to(device)
                cls_out, recon_out = model(s_tensor, f_tensor)
                loss, _, _ = criterion((cls_out, recon_out), (labels, f_tensor))
                val_loss += loss.item() * s_tensor.size(0)
                val_corrects += torch.sum(torch.max(cls_out, 1)[1] == labels.data)
        val_acc = val_corrects.double() / len(val_dataset)
        print(f'Val Loss: {val_loss / len(val_dataset):.4f} Acc: {val_acc:.4f}')

        if epoch >= warmup_epochs: scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc; torch.save(model.state_dict(), 'checkpoints/sffn_best_model.pth'); print("New best model saved!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the SFFN model."); parser.add_argument('--data_dir', required=True); parser.add_argument('--epochs', type=int, default=30); parser.add_argument('--batch_size', type=int, default=32); parser.add_argument('--lr', type=float, default=1e-4); parser.add_argument('--warmup_epochs', type=int, default=5)
    args = parser.parse_args()
    train_model(args.data_dir, args.epochs, args.batch_size, args.lr, args.warmup_epochs)