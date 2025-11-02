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


def calculate_dice_score(pred_mask, true_mask, num_classes=4):
    """Calculate Dice score for each class"""
    dice_scores = []
    class_names = ['Background', 'CSF', 'Gray Matter', 'White Matter']

    for class_id in range(num_classes):
        pred_class = (pred_mask == class_id)
        true_class = (true_mask == class_id)

        intersection = np.logical_and(pred_class, true_class).sum()
        union = pred_class.sum() + true_class.sum()

        dice = (2.0 * intersection) / (union + 1e-6) if union > 0 else 1.0
        dice_scores.append(dice)

    return dice_scores, class_names


def show_predictions(model, dataset, title="🧠 Brain Tissue Segmentation Results", n=5, save_path=None):
    """
    Show model predictions vs ground truth for multi-class segmentation.
    Includes Dice scores for each tissue class.
    """
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(16, 12))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.95)

    from matplotlib.colors import ListedColormap
    colors = ['black', 'blue', 'green', 'red']
    cmap = ListedColormap(colors)

    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]
            image_tensor = image.unsqueeze(0).to(device)

            # Model prediction
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred_mask = torch.argmax(probs, dim=0).cpu().numpy()
            true_mask_np = true_mask.squeeze().numpy()

            # Denormalize image for visualization
            img_show = denormalize_image(image)
            img_display = img_show.permute(1, 2, 0).numpy()

            # Calculate Dice scores
            dice_scores, class_names = calculate_dice_score(pred_mask, true_mask_np)

            # Plot original MRI
            axes[0, i].imshow(img_display, cmap='gray')
            axes[0, i].set_title(f'MRI Slice {i + 1}', fontweight='bold', fontsize=12)
            axes[0, i].axis('off')

            # Plot ground truth
            axes[1, i].imshow(true_mask_np, cmap=cmap, vmin=0, vmax=3)
            axes[1, i].set_title('Ground Truth', fontweight='bold', fontsize=12)
            axes[1, i].axis('off')

            # Plot prediction
            axes[2, i].imshow(pred_mask, cmap=cmap, vmin=0, vmax=3)

            # Add Dice scores as text
            dice_text = "\n".join([f"{name}: {score:.3f}" for name, score in zip(class_names, dice_scores)])
            avg_dice = np.mean(dice_scores)
            axes[2, i].set_title(f'Prediction\nAvg Dice: {avg_dice:.3f}', fontweight='bold', fontsize=12)
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
               bbox_to_anchor=(0.5, 0.02), fontsize=12, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.9)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"📊 Prediction visualization saved to {save_path}")
    else:
        plt.show()


model = load_trained_model()

# Get data
_, val_loader = get_dataloaders(batch_size=8, img_size=(256, 256))
val_dataset = val_loader.dataset

# Show predictions on sample images
print("🎯 Showing sample predictions...")
show_predictions(model, val_dataset, n=5, save_path="final_predictions.png")

