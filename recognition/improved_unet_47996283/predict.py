import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import get_dataloaders, denormalize_image, MRIDataset
from modules import UNet
import torchvision.transforms as transforms

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_trained_model(model_path='best_model.pth', num_classes=4):
    """Load the trained model from checkpoint"""
    model = UNet(in_channels=1, out_channels=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully from {model_path}")
    return model
