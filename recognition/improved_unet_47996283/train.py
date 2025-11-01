import torch
from dataset import get_dataloaders, denormalize_image
import matplotlib.pyplot as plt
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def show_epoch_predictions(model, dataset, epoch, n=3, denormalize_fn=denormalize_image):
    """
    Visualize model predictions after a specific epoch for multi-class segmentation.
    :param model: Trained UNet model.
    :param dataset: Dataset object (MRI + mask pairs).
    :param epoch: Current epoch number.
    :param n: Number of examples to display.
    :param denormalize_fn: Function to reverse normalization
    """
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(14, 10))
    fig.suptitle(f"MRI Segmentation Predictions After Epoch {epoch}", fontsize=16, fontweight='bold')

    with torch.no_grad():
        for i in range(n):
            # Load one sample
            image, true_mask = dataset[i]
            image = image.unsqueeze(0).to(device)  # Add batch dimension

            # Forward pass (softmax for multi-class)
            logits = model(image)
            probs = torch.softmax(logits, dim=1)[0]  # [C, H, W]
            pred_mask = torch.argmax(probs, dim=0).cpu().numpy()  # Class indices

            # Prepare ground truth
            true_mask_np = true_mask.numpy()

            # Denormalize for display
            img_show = denormalize_fn(image[0].cpu()) if denormalize_fn else image[0].cpu()
            img_display = img_show.permute(1, 2, 0).numpy()  # CHW -> HWC

            # Plot original image
            axes[0, i].imshow(img_display, cmap='gray')
            axes[0, i].set_title(f"Original {i + 1}", fontweight='bold')
            axes[0, i].axis('off')

            # Plot ground truth mask
            axes[1, i].imshow(true_mask_np, cmap='tab10', vmin=0, vmax=dataset.num_classes - 1)
            axes[1, i].set_title("Ground Truth", fontweight='bold')
            axes[1, i].axis('off')

            # Plot predicted mask
            axes[2, i].imshow(pred_mask, cmap='tab10', vmin=0, vmax=dataset.num_classes - 1)
            acc = np.mean(pred_mask == true_mask_np)
            axes[2, i].set_title(f"Prediction (Acc: {acc:.3f})", fontweight='bold')
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()
    model.train()  # Back to training mode


train_loader, val_loader = get_dataloaders(batch_size=8, img_size=(256, 256))

