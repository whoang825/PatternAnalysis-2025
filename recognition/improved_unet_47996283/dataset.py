import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


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


def show_examples(dataset, title="MRI Dataset Examples", n=3, save_path=None):
    """
    Displays example MRI slices and their corresponding segmentation masks
    from the given dataset.
    :param save_path: the image path to save the example dataset images
    :param dataset: instance of MRIDataset containing MRI images and masks
    :param title: title for the plot
    :param n: number of examples
    :return:
    """
    fig, axes = plt.subplots(2, n, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i in range(n):
        image, mask = dataset[i]

        # Denormalize the grayscale MRI image for visualization
        img_show = denormalize_image(image)

        # Convert to numpy array for plotting
        img_display = img_show.squeeze(0).numpy()  # [1,H,W] -> [H,W]
        mask_display = mask.squeeze(0).numpy()

        # Show MRI Image
        axes[0, i].imshow(img_display, cmap='gray')
        axes[0, i].set_title(f'MRI Slice {i + 1}', fontweight='bold')
        axes[0, i].axis('off')

        # Show Corresponding Mask
        axes[1, i].imshow(mask_display, cmap='Reds', vmin=0, vmax=1)
        axes[1, i].set_title(f'Segmentation Mask {i + 1}', fontweight='bold')
        axes[1, i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# Show examples of the MRI Dataset with their segmentation masks
tl, vl = get_dataloaders(batch_size=4)
td = tl.dataset
vd = vl.dataset
show_examples(td, title="MRI Training Dataset + Binary Masks", n=3, save_path="train_examples.png")
show_examples(vd, title="MRI Validation Dataset + Binary Masks", n=3, save_path="validation_examples.png")
