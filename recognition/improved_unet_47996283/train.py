import torch
from dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader, val_loader = get_dataloaders(batch_size=8, img_size=(256, 256))

