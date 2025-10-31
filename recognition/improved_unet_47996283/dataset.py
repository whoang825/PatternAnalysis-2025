import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class MRIDataset(Dataset):
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
