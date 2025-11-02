import torch
from dataset import get_dataloaders, denormalize_image
from modules import UNet, DiceLoss, CombinedLoss
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

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

    from matplotlib.colors import ListedColormap
    colors = ['black', 'blue', 'green', 'red']
    cmap = ListedColormap(colors)

    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]
            image = image.unsqueeze(0).to(device)

            logits = model(image)
            probs = torch.softmax(logits, dim=1)[0]
            pred_mask = torch.argmax(probs, dim=0).cpu().numpy()
            true_mask_np = true_mask.squeeze().numpy()

            img_show = denormalize_fn(image[0].cpu())
            img_display = img_show.permute(1, 2, 0).numpy()

            # Calculate per-class Dice scores
            dice_scores = []
            class_names = ['Background', 'CSF', 'Gray Matter', 'White Matter']
            for class_id in range(num_classes):
                pred_class = (pred_mask == class_id)
                true_class = (true_mask_np == class_id)

                intersection = np.logical_and(pred_class, true_class).sum()
                union = pred_class.sum() + true_class.sum()
                dice = (2.0 * intersection) / (union + 1e-6) if union > 0 else 1.0
                dice_scores.append(dice)

            avg_dice = np.mean(dice_scores)

            # Plot
            axes[0, i].imshow(img_display, cmap='gray')
            axes[0, i].set_title(f"MRI {i + 1}", fontweight='bold')
            axes[0, i].axis('off')

            axes[1, i].imshow(true_mask_np, cmap=cmap, vmin=0, vmax=num_classes - 1)
            axes[1, i].set_title("Ground Truth", fontweight='bold')
            axes[1, i].axis('off')

            axes[2, i].imshow(pred_mask, cmap=cmap, vmin=0, vmax=num_classes - 1)

            # Show individual class Dice scores
            dice_text = "\n".join([f"{class_names[c]}: {dice_scores[c]:.3f}" for c in range(num_classes)])
            axes[2, i].set_title(f"Prediction\nAvg Dice: {avg_dice:.3f}", fontweight='bold')
            axes[2, i].text(0.02, 0.98, dice_text, transform=axes[2, i].transAxes, fontsize=8,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
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
    plt.savefig(f"predictions_epoch{epoch}.png", bbox_inches='tight', dpi=150)
    model.train()


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


def get_class_weights(train_loader, num_classes=4, method='balanced'):
    """Calculate class weights based on frequency"""
    class_counts = torch.zeros(num_classes)

    print("Calculating class weights from training data...")

    # Count pixels for each class
    for images, masks in train_loader:
        masks_flat = masks.squeeze(1).flatten().long()
        for c in range(num_classes):
            class_counts[c] += (masks_flat == c).sum().item()

    print(f"Class counts: {class_counts}")
    print(f"Class percentages: {(class_counts / class_counts.sum() * 100).tolist()}")

    if method == 'balanced':
        # Standard balanced weighting (inverse frequency)
        total = class_counts.sum()
        weights = total / (num_classes * class_counts)
    elif method == 'sqrt':
        # Square root - less aggressive reweighting
        weights = 1.0 / torch.sqrt(class_counts / class_counts.sum())

    # Normalize weights
    weights = weights / weights.sum() * num_classes

    return weights.to(device)


def train(model, train_loader, val_loader, num_classes=4, epochs=100, lr=1e-4, visualize_every=10):
    """
    Train the U-Net model for multi-class MRI segmentation.
    :param model: U-Net model from modules.py
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for visualization/testing
    :param num_classes: Number of segmentation classes
    :param epochs: Number of training epochs
    :param lr: Learning rate
    :param visualize_every: Interval (in epochs) to visualize predictions
    """
    model.to(device)

    class_weights = get_class_weights(train_loader, num_classes)
    print(f"Using class weights: {class_weights}")

    criterion = CombinedLoss(num_classes=num_classes, class_weights=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    losses = []
    best_dice = 0.0

    print(f"Starting Training with {num_classes}-Class Segmentation...")

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

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # Calculate validation Dice score
        val_dice = evaluate_dice(model, val_loader, num_classes)

        # Print learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1} — LR: {current_lr:.2e} | Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_model.pth')

        # Visualize predictions
        if (epoch + 1) % visualize_every == 0:
            show_epoch_predictions(model, val_loader.dataset, epoch + 1, n=3, num_classes=num_classes)

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    print(f"Training complete! Best validation Dice: {best_dice:.4f}")

    return losses


def evaluate_dice(model, val_loader, num_classes=4):
    """Calculate Dice score on validation set"""
    model.eval()
    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            targets = masks.squeeze(1).long()

            batch_dice = 0.0
            for c in range(num_classes):
                pred_c = (preds == c).float()
                target_c = (targets == c).float()

                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()

                dice = (2.0 * intersection) / (union + 1e-6)
                batch_dice += dice.item()

            total_dice += batch_dice / num_classes
            num_batches += 1

    model.train()
    return total_dice / num_batches if num_batches > 0 else 0.0


train_loader, val_loader = get_dataloaders(batch_size=8, img_size=(256, 256))
model = UNet(in_channels=1, out_channels=4)
train_losses = train(model, train_loader, val_loader, num_classes=4, epochs=100, lr=1e-4, visualize_every=10)
plot_loss(train_losses, metric_name="Dice Loss")

