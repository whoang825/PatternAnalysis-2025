import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    U-Net with BatchNorm, LeakyReLU, Dropout, and Sigmoid activation:
    CNN for MRI Segmentation, outputting a segmentation mask representing the probability
    that the pixel belongs to the target region (e.g. brain tissue, tumor).
    """
    def __init__(self, in_channels=1, out_channels=4, base_filters=32, dropout_p=0.3):
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
        """Upsampling block using ConvTranspose2d for learnable upsampling to reduce channels.
        Restore the spatial resolution of the feature maps to localize features precisely.
        """
        # Improves reconstruction accuracy and feature recovery
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        """Forward pass of the Improved U-Net.
        Skip connections is used to reconstruct low-level features from the encoder
        """
        # Encoder (extract the deepest features)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck (global features of the image)
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        # Upsampling feature map for segmentation
        d4 = self.up4(b)
        # Concatenates the corresponding encoder output to compare the high-res details with
        # low-res details to reconstruct low-level features from the encoder
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Final output (B, 4, H, W)
        out = self.final_conv(d1)

        return out


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation:
    Minimizing Dice Loss = Maximising Dice Coefficient (how well two sets overlap).
    Dice Loss = 1 - Dice Coefficient.
    Dice Coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)
    """

    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        predictions: raw logits [B, C, H, W]
        targets: integer labels [B, H, W] (0..C-1)
        """
        # Convert to probabilities
        probs = F.softmax(predictions, dim=1)

        dice_loss = 0
        for c in range(self.num_classes):
            # Create binary mask for class c
            # Isolate predictions and targets for each class
            pred_c = probs[:, c, :, :]
            target_c = (targets == c).float()
            # Compute intersection and Dice score
            intersection = (pred_c * target_c).sum()
            dice_c = (2. * intersection + self.smooth) / (pred_c.sum() + target_c.sum() + self.smooth)
            dice_loss += 1 - dice_c
        # Compute average of dice loss across each class
        return dice_loss / self.num_classes

