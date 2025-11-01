import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    U-Net with BatchNorm, LeakyReLU, Dropout, and Sigmoid activation:
    CNN for MRI Segmentation, outputting a segmentation mask representing the probability
    that the pixel belongs to the target region (e.g. brain tissue, tumor)
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, dropout_p=0.3):
        super().__init__()

    def _conv_block(self, in_ch, out_ch, dropout_p=0.3):
        """Conv block: Conv -> BN -> LeakyReLU -> Dropout -> Conv -> BN -> LeakyReLU -> Dropout
        To extract features from the input image.
        First layer extracts basic features like edges and corners.
        Second layer extracts more complex, high-level features.
        """
        return nn.Sequential(
            # padding=1 ensures output size remains same
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_p)
        )
