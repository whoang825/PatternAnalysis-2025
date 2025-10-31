import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class MRIDataset(Dataset):
    """
    Dataset for loading MRI images and their corresponding segmentation masks from
    the given directories.
    """
    def __init__(self, img_dir, masks_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        self.masks_files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]

        assert len(self.img_files) == len(self.masks_files), \
            f"Image and mask count mismatch: {len(self.img_files)} vs {len(self.masks_files)}"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Get paths
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.masks_dir, self.masks_files[idx])

        # Load images (convert both to grayscale)
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Apply transforms (same spatial transform for both)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask


def get_dataloaders(batch_size=8, img_size=(256, 256)):
    """
    Loads the datasets from the directory into the loaders for training.
    :param batch_size: number of samples per batch
    :param img_size: size of all images and masks (H,W)
    :return: both data loaders for the training and validation.
    """
    train_dir = "/home/groups/comp3710/OASIS/keras_png_slices_train"
    train_masks_dir = "/home/groups/comp3710/OASIS/keras_png_slices_seg_train"
    val_dir = "/home/groups/comp3710/OASIS/keras_png_slices_validate"
    val_masks_dir = "/home/groups/comp3710/OASIS/keras_png_slices_seg_validate"

    # Normalise the images for more stable training and resize the images and masks
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),

    ])

    train_dataset = MRIDataset(
        train_dir, train_masks_dir, transform=transform, target_transform=mask_transform
    )
    val_dataset = MRIDataset(
        val_dir, val_masks_dir, transform=transform, target_transform=mask_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader


def denormalize_image(tensor):
    """
    Undo normalisation for the grayscale MRI image for visualisation.
    :param tensor: tensor of shape [C,H,W] with values normalized to [-1,1]
    :return: tensor with values in [0,1]
    """
    mean = 0.5
    std = 0.5
    # Denormalize the tensor
    tensor = tensor * std + mean
    # Ensure the values stay between [0,1]
    return torch.clamp(tensor, 0, 1)
