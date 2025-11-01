import torch
from dataset import get_dataloaders, denormalize_image
from modules import UNet, DiceLoss
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def show_epoch_predictions(model, dataset, epoch, n=3, num_classes=4, denormalize_fn=denormalize_image):
    """
    Visualize model predictions after a specific epoch for multi-class segmentation.
    :param num_classes: number of classes for the image segmentation.
    :param model: Trained UNet model.
    :param dataset: Dataset object (MRI + mask pairs).
    :param epoch: Current epoch number.
    :param n: Number of examples to display.
    :param denormalize_fn: Function to reverse normalization
    """
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(14, 10))
    fig.suptitle(f"Brain Tissue Segmentation - Epoch {epoch}", fontsize=16, fontweight='bold')

    # Define colors for each class
    from matplotlib.colors import ListedColormap
    colors = ['black', 'blue', 'green', 'red']  # Background, CSF, Gray Matter, White Matter
    cmap = ListedColormap(colors)

    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]
            image = image.unsqueeze(0).to(device)

            # Forward pass (softmax for multi-class)
            logits = model(image)
            probs = torch.softmax(logits, dim=1)[0]  # [4, H, W]
            pred_mask = torch.argmax(probs, dim=0).cpu().numpy()  # [H, W]

            # Prepare ground truth
            true_mask_np = true_mask.squeeze().numpy()

            # Denormalize for display
            img_show = denormalize_fn(image[0].cpu())
            img_display = img_show.permute(1, 2, 0).numpy()

            # Plot original image
            axes[0, i].imshow(img_display, cmap='gray')
            axes[0, i].set_title(f"MRI {i + 1}", fontweight='bold')
            axes[0, i].axis('off')

            # Plot ground truth mask
            axes[1, i].imshow(true_mask_np, cmap=cmap, vmin=0, vmax=num_classes - 1)
            axes[1, i].set_title("Ground Truth", fontweight='bold')
            axes[1, i].axis('off')

            # Plot predicted mask
            axes[2, i].imshow(pred_mask, cmap=cmap, vmin=0, vmax=num_classes - 1)
            axes[2, i].set_title(f"Prediction Masks", fontweight='bold')
            axes[2, i].axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', label='Background'),
        Patch(facecolor='blue', label='CSF'),
        Patch(facecolor='green', label='Gray Matter'),
        Patch(facecolor='red', label='White Matter')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.02), fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"predictions_epoch{epoch}.png", bbox_inches='tight')
    model.train()  # Back to training mode


def plot_loss(train_losses, val_losses=None, metric_name='Dice Coefficient'):
    """
    Plot training and validation loss/metric curves.
    :param train_losses: List of training loss values (or metrics).
    :param val_losses: List of validation loss/metric values.
    :param metric_name: Display name for the y-axis.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, 'bo-', label='Training', linewidth=2, markersize=6)
    if val_losses is not None:
        plt.plot(val_losses, 'ro-', label='Validation', linewidth=2, markersize=6)

    plt.title(f"{metric_name} Over Epochs", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("loss_plot.png")


def train(model, train_loader, val_dataset, num_classes=4, epochs=20, lr=1e-3, visualize_every=5):
    """
    Train the U-Net model for multi-class MRI segmentation.
    :param model: U-Net model from modules.py
    :param train_loader: DataLoader for training data
    :param val_dataset: Dataset for visualization/testing
    :param num_classes: Number of segmentation classes
    :param epochs: Number of training epochs
    :param lr: Learning rate
    :param visualize_every: Interval (in epochs) to visualize predictions
    """
    model.to(device)
    criterion = DiceLoss(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    print(f"Starting Training with {num_classes}-Class Segmentation (Softmax + Dice Loss)...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Training loop
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Print periodic updates (every 200 batches)
            if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} — Average Loss: {avg_loss:.4f}")

        # Visualize predictions every few epochs
        if (epoch + 1) % visualize_every == 0:
            show_epoch_predictions(model, val_dataset, epoch + 1, n=3, num_classes=num_classes)

    print("Training Complete!")
    return train_losses


train_loader, val_loader = get_dataloaders(batch_size=8, img_size=(256, 256))
model = UNet(in_channels=1, out_channels=4)
train_losses = train(model, train_loader, val_loader.dataset, num_classes=4, epochs=20, lr=1e-3)
plot_loss(train_losses, metric_name="Dice Loss")

