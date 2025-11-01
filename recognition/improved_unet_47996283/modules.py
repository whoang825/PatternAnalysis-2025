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

        # Encoder (Downsampling)
        # Captures context by reducing spatial resolution
        self.enc1 = self._conv_block(in_channels, base_filters, dropout_p)  # 256 x 256
        self.enc2 = self._conv_block(base_filters, base_filters * 2, dropout_p)  # 128 x 128
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4, dropout_p)  # 64 x 64
        self.enc4 = self._conv_block(base_filters * 4, base_filters * 8, dropout_p)  # 32 x 32

        # Bottleneck
        # Deepest layers capturing abstract features
        self.bottleneck = self._conv_block(base_filters * 8, base_filters * 16, dropout_p)

        # Decoder (Upsampling)
        self.up4 = self._up_block(base_filters * 16, base_filters * 8)
        self.dec4 = self._conv_block(base_filters * 16, base_filters * 8, dropout_p)

        self.up3 = self._up_block(base_filters * 8, base_filters * 4)
        self.dec3 = self._conv_block(base_filters * 8, base_filters * 4, dropout_p)

        self.up2 = self._up_block(base_filters * 4, base_filters * 2)
        self.dec2 = self._conv_block(base_filters * 4, base_filters * 2, dropout_p)

        self.up1 = self._up_block(base_filters * 2, base_filters)
        self.dec1 = self._conv_block(base_filters * 2, base_filters, dropout_p)

        # Final output layer
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Pooling for encoder
        self.pool = nn.MaxPool2d(2, 2)

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

    def _up_block(self, in_ch, out_ch):
        """Upsample using bilinear interpolation followed by a 1x1 conv to reduce channels.
        Restore the spatial resolution of the feature maps to localize features precisely.
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
        )

